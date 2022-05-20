struct LiftedTrajectoryGameSolver{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10}
    "A collection of action generators, one for each player in the game."
    trajectory_reference_generators::T1
    "A collection of trajectory generators, one for each player in the game"
    trajectory_generators::T2
    "A callable `(game, xs, us) -> cs` which maps the joint state-input trajectory `(xs, us)` to a
    tuple of scalar costs `cs` for a given `game`. In the simplest case, this may just forward to
    the `game.cost`. More generally, however, this function will add penalties to enforce
    constraints."
    coupling_constraints_handler::T3
    "The number of time steps to plan into the future."
    planning_horizon::T4
    "A random number generator to generate non-deterministic strategies."
    rng::T5
    "How much affect the dual regularization has on the costs"
    dual_regularization_weights::T6
    "A flag that can be set to enable/disable learning"
    enable_learning::T7
    "An AbstractExecutionPolicy that determines whether the solve is computed in parallel or
    sequentially."
    execution_policy::T8
    "A state value predictor (e.g. a neural network) that maps the current state to a tuple of
    optimal cost-to-go's for each player."
    state_value_predictor::T9
    "A function to compose the input of the reference generator from parameters (player_i, state, context)."
    compose_reference_generator_input::T10
end

"""
Convenience constructor to derive a suitable solver directly from a given game.
"""
function LiftedTrajectoryGameSolver(
    game::TrajectoryGame{<:ProductDynamics},
    planning_horizon;
    trajectory_parameterizations = [
        InputReferenceParameterization(; α = 3) for _ in 1:num_players(game)
    ],
    trajectory_generator_constructors = [
        DifferentiableTrajectoryGenerator for _ in 1:num_players(game)
    ],
    rng = Random.MersenneTwister(1),
    context_dimension = 0,
    reference_generator_input_dimension = state_dim(game.dynamics) + context_dimension,
    initial_parameters = [:random for _ in 1:num_players(game)],
    n_actions = [2 for _ in 1:num_players(game)],
    learning_rates = [0.05 for _ in 1:num_players(game)],
    reference_generator_constructors = [NNActionGenerator for _ in 1:num_players(game)],
    gradient_clipping_threshold = nothing,
    coupling_constraints_handler = LagrangianCouplingConstraintHandler(100),
    dual_regularization_weights = [1e-4 for _ in 1:num_players(game)],
    enable_learning = [true for _ in 1:num_players(game)],
    execution_policy = SequentialExecutionPolicy(),
    state_value_predictor = nothing,
    compose_reference_generator_input = (i, game_state, context) -> [game_state; context],
)
    trajectory_generators = map(
        trajectory_generator_constructors,
        game.dynamics.subsystems,
        trajectory_parameterizations,
    ) do constructor, subdynamics, parameterization
        constructor(
            game.env,
            subdynamics,
            state_dim(subdynamics),
            control_dim(subdynamics),
            parameterization,
            planning_horizon,
        )
    end

    if execution_policy isa MultiThreadedExecutionPolicy &&
       any(!is_thread_safe, trajectory_generators)
        throw(
            ArgumentError(
                """
                The solver trajectory optimization backend that you selected does not support \
                multi-threaded execution. Consider using a another backend or disable \
                multi-threading by handing another `execution_policy`.
                """,
            ),
        )
    end

    trajectory_reference_generators = map(
        reference_generator_constructors,
        trajectory_generators,
        n_actions,
        initial_parameters,
        learning_rates,
    ) do constructor, trajectory_generator, n_actions, initial_parameters, learning_rate
        constructor(;
            input_dimension = reference_generator_input_dimension,
            parameter_dimension = parameter_dimension(trajectory_generator),
            n_actions,
            learning_rate,
            rng,
            initial_parameters,
            gradient_clipping_threshold,
        )
    end
    LiftedTrajectoryGameSolver(
        trajectory_reference_generators,
        trajectory_generators,
        coupling_constraints_handler,
        planning_horizon,
        rng,
        dual_regularization_weights,
        enable_learning,
        execution_policy,
        state_value_predictor,
        compose_reference_generator_input,
    )
end

# π
function generate_trajectory_references(solver, initial_state, context_state; n_players)
    map(1:n_players) do player_i
        input = solver.compose_reference_generator_input(player_i, initial_state, context_state)
        solver.trajectory_reference_generators[player_i](input)
    end
end

# TRAJ
function generate_trajectories(
    solver,
    initial_state,
    stacked_references;
    n_players,
    trajectory_pairings,
)
    state_per_player = blocks(initial_state)
    references_per_player = let
        iterable_references = Iterators.Stateful(eachcol(stacked_references))
        map(1:n_players) do ii
            n_references = size(trajectory_pairings, ii)
            collect(Iterators.take(iterable_references, n_references))
        end
    end

    map_threadable(1:n_players, solver.execution_policy) do player_i
        references = references_per_player[player_i]
        trajectory_generator = solver.trajectory_generators[player_i]
        substate = state_per_player[player_i]
        map(reference -> trajectory_generator(substate, reference), references)
    end
end

function compute_costs(
    solver,
    trajectories_per_player,
    context_state;
    trajectory_pairings,
    n_players,
    game,
)
    cost_tensor = map_threadable(trajectory_pairings, solver.execution_policy) do i
        trajectories = (trajectories_per_player[j][i[j]] for j in 1:n_players)

        cost_horizon =
            isnothing(solver.state_value_predictor) ? solver.planning_horizon + 1 :
            solver.state_value_predictor.turn_length

        xs = map((t.xs[1:cost_horizon] for t in trajectories)...) do x...
            mortar(collect(x))
        end

        us = map((t.us[1:(cost_horizon - 1)] for t in trajectories)...) do u...
            mortar(collect(u))
        end

        trajectory_costs = game.cost(xs, us, context_state)
        if !isnothing(game.coupling_constraints)
            trajectory_costs .+= solver.coupling_constraints_handler(game, xs, us, context_state)
        end

        if !isnothing(solver.state_value_predictor)
            trajectory_costs .+=
                game.cost.discount_factor .* solver.state_value_predictor(xs[cost_horizon])
        end

        trajectory_costs
    end

    # transpose tensor of tuples to tuple of tensors
    map(1:n_players) do player_i
        map(cost_tensor) do pairing
            pairing[player_i]
        end
    end
end

function compute_regularized_loss(
    trajectories_per_player,
    game_value_per_player;
    n_players,
    planning_horizon,
    dual_regularization_weights,
)
    function huber(x; δ = 1)
        if abs(x) > δ
            δ * (abs(x) - 0.5δ)
        else
            0.5x^2
        end
    end

    dual_regularizations =
        [sum(sum(huber.(c.λs)) for c in trajectories_per_player[i]) for i in 1:n_players] ./ planning_horizon

    loss_per_player = [
        game_value_per_player[i] + dual_regularization_weights[i] * dual_regularizations[i] for
        i in 1:n_players
    ]
end

# Make this compatable with many players
function forward_pass(; solver, game, initial_state, context_state, min_action_probability)
    n_players = num_players(game)
    references_per_player =
        generate_trajectory_references(solver, initial_state, context_state; n_players)

    stacked_references = reduce(hcat, references_per_player)

    local trajectories_per_player, mixing_strategies, game_value_per_player

    loss_per_player = Zygote.forwarddiff(
        stacked_references;
        chunk_threshold = length(stacked_references),
    ) do stacked_references
        trajectory_pairings =
            Iterators.product([axes(references, 2) for references in references_per_player]...) |> collect
        trajectories_per_player = generate_trajectories(
            solver,
            initial_state,
            stacked_references;
            n_players,
            trajectory_pairings,
        )
        # f
        # Evaluate the functions on all joint trajectories in the cost tensor
        cost_tensors_per_player = compute_costs(
            solver,
            trajectories_per_player,
            context_state;
            trajectory_pairings,
            n_players,
            game,
        )
        # Compute the mixing strategies, q_i, via a finite game solve;
        mixing_strategies =
            TensorGames.compute_equilibrium(cost_tensors_per_player; ϵ = min_action_probability).x
        # L
        game_value_per_player = [
            TensorGames.expected_cost(mixing_strategies, cost_tensor) for
            cost_tensor in cost_tensors_per_player
        ]

        compute_regularized_loss(
            trajectories_per_player,
            game_value_per_player;
            n_players,
            solver.planning_horizon,
            solver.dual_regularization_weights,
        )
    end

    # strip of dual number types for downstream operation
    info = clean_info_tuple(; game_value_per_player, mixing_strategies, trajectories_per_player)

    (; loss_per_player, info)
end

function clean_info_tuple(; game_value_per_player, mixing_strategies, trajectories_per_player)
    (;
        game_value_per_player = ForwardDiff.value.(game_value_per_player),
        mixing_strategies = [ForwardDiff.value.(q) for q in mixing_strategies],
        trajectories_per_player = map(trajectories_per_player) do trajectories
            map(trajectories) do trajectory
                (;
                    xs = [ForwardDiff.value.(x) for x in trajectory.xs],
                    us = [ForwardDiff.value.(u) for u in trajectory.us],
                    λs = ForwardDiff.value.(trajectory.λs),
                )
            end
        end,
    )
end

function cost_gradients(back, solver, game)
    cost_gradients(back, solver, game, game.cost.structure)
end

function cost_gradients(back, solver, game, ::GeneralSumCostStructure)
    n_players = num_players(game)
    ∇L = map(1:n_players) do n
        if solver.enable_learning[n]
            loss_per_player = [i == n ? 1 : 0 for i in 1:n_players]
            back((; loss_per_player, info = nothing)) |> copy
        else
            nothing
        end
    end
end

function cost_gradients(back, solver, game, ::ZeroSumCostStructure)
    num_players(game) == 2 || error("Not implemented for N>2 players")
    if !isnothing(game.coupling_constraints)
        return cost_gradients(back, solver, game, GeneralSumCostStructure())
    end
    if !any(solver.enable_learning)
        return ∇L = (nothing, nothing)
    end
    ∇L_1 = back((; loss_per_player = [1, 0], info = nothing))
    ∇L = [∇L_1, -1 .* ∇L_1]
end

function update_state_value_predictor!(solver, state, game_value_per_player)
    push!(
        solver.state_value_predictor.replay_buffer,
        (; value_target_per_player = game_value_per_player, state),
    )

    if length(solver.state_value_predictor.replay_buffer) >= solver.state_value_predictor.batch_size
        fit_value_predictor!(solver.state_value_predictor)
        empty!(solver.state_value_predictor.replay_buffer)
    end
end

# TODO: move parameters to solver
function TrajectoryGamesBase.solve_trajectory_game!(
    solver::LiftedTrajectoryGameSolver,
    game::TrajectoryGame{<:ProductDynamics},
    initial_state;
    context_state = Float64[],
    min_action_probability = 0.05,
    parameter_noise = 0.0,
    scale_action_gradients = true,
)
    n_players = num_players(game)
    if !isnothing(solver.enable_learning) && any(solver.enable_learning)
        trainable_parameters = Flux.params(solver.trajectory_reference_generators...)
        forward_pass_result, back = Zygote.pullback(
            () -> forward_pass(;
                solver,
                game,
                initial_state,
                context_state,
                min_action_probability,
            ),
            trainable_parameters,
        )
        ∇L_per_player = cost_gradients(back, solver, game)
    else
        forward_pass_result =
            forward_pass(; solver, game, initial_state, context_state, min_action_probability)
        ∇L_per_player = [nothing for _ in 1:n_players]
    end

    (; loss_per_player, info) = forward_pass_result

    # Update θ_i if learning is enabled for player i
    if !isnothing(solver.enable_learning)
        for (reference_generator, weights, enable_player_learning, ∇L) in zip(
            solver.trajectory_reference_generators,
            info.mixing_strategies,
            solver.enable_learning,
            ∇L_per_player,
        )
            if !enable_player_learning
                continue
            end
            action_gradient_scaling = scale_action_gradients ? 1 ./ weights : ones(size(weights))
            update_parameters!(
                reference_generator,
                ∇L;
                noise = parameter_noise,
                solver.rng,
                action_gradient_scaling,
            )
        end
    end

    if !isnothing(solver.state_value_predictor) &&
       !isnothing(solver.enable_learning) &&
       any(solver.enable_learning)
        update_state_value_predictor!(solver, initial_state, info.game_value_per_player)
    end

    γs = map(
        1:n_players,
        info.mixing_strategies,
        loss_per_player,
        info.trajectories_per_player,
        ∇L_per_player,
    ) do player_i, weights, loss, trajectories, ∇L
        ∇L_norm = if isnothing(∇L)
            0.0
        else
            sum(
                norm(something(∇L[p], 0.0)) for
                p in Flux.params(solver.trajectory_reference_generators...)
            )
        end
        LiftedTrajectoryStrategy(;
            player_i,
            trajectories,
            weights,
            info = (; loss, ∇L_norm),
            solver.rng,
        )
    end

    JointStrategy(γs)
end
