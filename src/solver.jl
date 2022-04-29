struct LiftedTrajectoryGameSolver{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10}
    "A collection of action generators, one for each player in the game."
    trajectory_parameter_generators::T1
    "A acollection of trajectory generators, one for each player in the game"
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
    "A vector of cached trajectories for each player"
    trajectory_caches::T8
    "An AbstractExecutionPolicy that determines whether the solve is computed in parallel or
    sequentially."
    execution_policy::T9
    "A state value predictor (e.g. a neural network) that maps the current state to a tuple of
    optimal cost-to-go's for each player."
    state_value_predictor::T10
end

"""
Convenience constructor to drive a suitable solver directly form a given game.
"""
function LiftedTrajectoryGameSolver(
    game::TrajectoryGame{<:ProductDynamics},
    planning_horizon;
    rng = Random.MersenneTwister(1),
    initial_parameters = [:random for _ in 1:num_players(game)],
    n_actions = [2 for _ in 1:num_players(game)],
    reference_generator_constructors = [NNActionGenerator for _ in 1:num_players(game)],
    learning_rates = [0.05 for _ in 1:num_players(game)],
    trajectory_parameterizations = [
        InputReferenceParameterization(; α = 3, params_abs_max = 10) for _ in 1:num_players(game)
    ],
    coupling_constraints_handler = LangrangianCouplingConstraintHandler(100),
    trajectory_solver = QPSolver(),
    dual_regularization_weights = [1e-4 for _ in 1:num_players(game)],
    enable_learning = [true for _ in 1:num_players(game)],
    trajectory_caches = [nothing for _ in 1:num_players(game)],
    gradient_clipping_threshold = nothing,
    execution_policy = SequentialExecutionPolicy(),
    state_value_predictor = nothing,
)
    # setup a trajectory generator for every player
    trajectory_generators =
        map(game.dynamics.subsystems, trajectory_parameterizations) do subdynamics, parameterization
            DifferentiableTrajectoryGenerator(
                ParametricOptimizationProblem(
                    parameterization,
                    subdynamics,
                    game.env,
                    planning_horizon,
                ),
                trajectory_solver,
            )
        end
    trajectory_parameter_generators = map(
        reference_generator_constructors,
        trajectory_generators,
        n_actions,
        initial_parameters,
        learning_rates,
    ) do constructor,
    trajectory_generator,
    n_player_actions,
    initial_player_parameters,
    learning_rate
        constructor(;
            state_dim = state_dim(game.dynamics),
            n_params = param_dim(trajectory_generator),
            n_actions = n_player_actions,
            trajectory_generator.problem.parameterization.params_abs_max,
            learning_rate,
            rng,
            initial_parameters = initial_player_parameters,
            gradient_clipping_threshold,
        )
    end

    LiftedTrajectoryGameSolver(
        trajectory_parameter_generators,
        trajectory_generators,
        coupling_constraints_handler,
        planning_horizon,
        rng,
        dual_regularization_weights,
        enable_learning,
        trajectory_caches,
        execution_policy,
        state_value_predictor,
    )
end

function huber(x; δ = 1)
    if abs(x) > δ
        δ * (abs(x) - 0.5δ)
    else
        0.5x^2
    end
end

# π
function generate_trajectory_references(solver, initial_state; enable_caching_per_player)
    state_per_player = blocks(initial_state)
    n_players = length(state_per_player)
    references_per_player = map_threadable(1:n_players, solver.execution_policy) do ii
        cache = solver.trajectory_caches[ii]
        if !isnothing(cache) && enable_caching_per_player[ii]
            cache
        else
            # π_θi
            references = solver.trajectory_parameter_generators[ii](initial_state)
        end
    end
    references_per_player
end

# TRAJ
function generate_trajectory_candidates(
    solver,
    initial_state,
    stacked_references;
    n_players,
    trajectory_pairings,
)
    state_per_player = blocks(initial_state)
    iterable_references = Iterators.Stateful(eachcol(stacked_references))

    map(1:n_players) do ii
        n_references = size(trajectory_pairings, ii)
        references = collect(Iterators.take(iterable_references, n_references))
        trajectory_generator = solver.trajectory_generators[ii]
        substate = state_per_player[ii]
        map_threadable(
            reference -> (; trajectory = trajectory_generator(substate, reference), reference),
            references,
            solver.execution_policy,
        )
    end
end

function compute_costs(solver, candidates_per_player; trajectory_pairings, n_players, game)
    cost_tensor = map_threadable(trajectory_pairings, solver.execution_policy) do i
        trajectories = (candidates_per_player[j][i[j]].trajectory for j in 1:n_players)

        xs = map((t.xs for t in trajectories)...) do x...
            mortar(collect(x))
        end

        us = map((t.us for t in trajectories)...) do u...
            mortar(collect(u))
        end

        turn_length =
            isnothing(solver.state_value_predictor) ? length(xs) :
            solver.state_value_predictor.turn_length

        trajectory_cost = solver.coupling_constraints_handler(
            game,
            xs[1:turn_length],
            us[1:(turn_length - 1)],
        )

        if isnothing(solver.state_value_predictor)
            cost_to_go = 0
        else
            # TODO: in the zero-sum case we could have a specialized state_value_predictor that
            # exploits the symmetry in the state value.
            cost_to_go = game.cost.discount_factor .* solver.state_value_predictor(xs[turn_length])
        end

        trajectory_cost .+ cost_to_go
    end

    # transpose tensor of tuples to tuple of tensors
    map(1:n_players) do player_i
        map(cost_tensor) do pairing
            pairing[player_i]
        end
    end
end

function compute_regularized_loss(
    candidates_per_player,
    game_value_per_player;
    n_players,
    planning_horizon,
    dual_regularization_weights,
)
    dual_regularizations =
        [
            sum(sum(huber.(c.trajectory.λs)) for c in candidates_per_player[i]) for i in 1:n_players
        ] ./ planning_horizon

    loss_per_player = [
        game_value_per_player[i] + dual_regularization_weights[i] * dual_regularizations[i] for
        i in 1:n_players
    ]
end

# Make this compatable with many players
function forward_pass(;
    solver,
    game,
    initial_state,
    min_action_probability,
    enable_caching_per_player,
)
    n_players = num_players(game)
    references_per_player =
        generate_trajectory_references(solver, initial_state; enable_caching_per_player)

    stacked_references = reduce(hcat, references_per_player)

    local candidates_per_player, mixing_strategies, game_value_per_player

    loss_per_player = Zygote.forwarddiff(
        stacked_references;
        chunk_threshold = length(stacked_references),
    ) do stacked_references
        trajectory_pairings =
            Iterators.product([axes(references, 2) for references in references_per_player]...) |> collect
        candidates_per_player = generate_trajectory_candidates(
            solver,
            initial_state,
            stacked_references;
            n_players,
            trajectory_pairings,
        )
        # f
        # Evaluate the functions on all joint trajectories in the cost tensor
        cost_tensors_per_player =
            compute_costs(solver, candidates_per_player; trajectory_pairings, n_players, game)
        # Compute the mixing strategies, q_i, via a finite game solve;
        mixing_strategies =
            TensorGames.compute_equilibrium(cost_tensors_per_player; ϵ = min_action_probability).x
        # L
        game_value_per_player = [
            TensorGames.expected_cost(mixing_strategies, cost_tensor) for
            cost_tensor in cost_tensors_per_player
        ]

        compute_regularized_loss(
            candidates_per_player,
            game_value_per_player;
            n_players,
            solver.planning_horizon,
            solver.dual_regularization_weights,
        )
    end

    # strip of dual number types for downstream operation
    info = clean_info_tuple(; game_value_per_player, mixing_strategies, candidates_per_player)

    (; loss_per_player, info)
end

function clean_info_tuple(; game_value_per_player, mixing_strategies, candidates_per_player)
    (;
        game_value_per_player = ForwardDiff.value.(game_value_per_player),
        mixing_strategies = [ForwardDiff.value.(q) for q in mixing_strategies],
        candidates_per_player = map(candidates_per_player) do candidates
            map(candidates) do candidate
                (;
                    reference = ForwardDiff.value.(candidate.reference),
                    trajectory = (;
                        xs = [ForwardDiff.value.(x) for x in candidate.trajectory.xs],
                        us = [ForwardDiff.value.(u) for u in candidate.trajectory.us],
                        λs = ForwardDiff.value.(candidate.trajectory.λs),
                    ),
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
    if isnothing(game.coupling_constraints)
        return cost_gradients(back, solver, game, GeneralSumCostStructure())
    end
    if !any(solver.enable_learning)
        return ∇L = (nothing, nothing)
    end
    ∇L_1 = back((; loss_per_player = [1, 0], info = nothing))
    ∇L = (∇L_1, -1 .* ∇L_1)
end

function update_state_value_predictor!(solver, state, game_value_per_player)
    if !isnothing(solver.state_value_predictor) &&
       !isnothing(solver.enable_learning) &&
       any(solver.enable_learning)
        push!(
            solver.state_value_predictor.replay_buffer,
            (; value_target_per_player = game_value_per_player, state),
        )

        if length(solver.state_value_predictor.replay_buffer) >=
           solver.state_value_predictor.batch_size
            fit_value_predictor!(solver.state_value_predictor)
            empty!(solver.state_value_predictor.replay_buffer)
        end
    end
end

function TrajectoryGamesBase.solve_trajectory_game!(
    solver::LiftedTrajectoryGameSolver,
    game::TrajectoryGame{<:ProductDynamics},
    initial_state;
    min_action_probability = 0.05,
    enable_caching_per_player = zeros(Bool, num_players(game)),
    parameter_noise = 0.0,
    scale_action_gradients = true,
)
    n_players = num_players(game)
    if !isnothing(solver.enable_learning) && any(solver.enable_learning)
        trainable_parameters = Flux.params(solver.trajectory_parameter_generators...)
        forward_pass_result, back = Zygote.pullback(
            () -> forward_pass(;
                solver,
                game,
                initial_state,
                min_action_probability,
                enable_caching_per_player,
            ),
            trainable_parameters,
        )
        ∇L_per_player = cost_gradients(back, solver, game)
    else
        forward_pass_result = forward_pass(;
            solver,
            game,
            initial_state,
            min_action_probability,
            enable_caching_per_player,
        )
        ∇L_per_player = [nothing for n in 1:n_players]
    end

    (; loss_per_player, info) = forward_pass_result

    # Store computed trajectories in caches if caching is enabled
    if !(eltype(solver.trajectory_caches) <: Nothing)
        for ii in eachindex(info.candidates_per_player)
            if enable_caching_per_player[ii]
                solver.trajectory_caches[ii] =
                    reduce(hcat, c.reference for c in info.candidates_per_player[ii])
            end
        end
    end

    # Update θ_i if learning is enabled for player i
    if !isnothing(solver.enable_learning)
        for (parameter_generator, weights, enable_player_learning, ∇L) in zip(
            solver.trajectory_parameter_generators,
            info.mixing_strategies,
            solver.enable_learning,
            ∇L_per_player,
        )
            if !enable_player_learning
                continue
            end
            action_gradient_scaling = scale_action_gradients ? 1 ./ weights : ones(size(weights))
            update_parameters!(
                parameter_generator,
                ∇L;
                noise = parameter_noise,
                solver.rng,
                action_gradient_scaling,
            )
        end
    end

    update_state_value_predictor!(solver, initial_state, info.game_value_per_player)

    γs = map(
        Iterators.countfrom(),
        info.mixing_strategies,
        loss_per_player,
        info.candidates_per_player,
        ∇L_per_player,
    ) do player_i, weights, loss, candidates, ∇L
        ∇L_norm = if isnothing(∇L)
            0.0
        else
            sum(
                norm(something(∇L[p], 0.0)) for
                p in Flux.params(solver.trajectory_parameter_generators...)
            )
        end
        LiftedTrajectoryStrategy(;
            player_i,
            trajectory_candidates = [c.trajectory for c in candidates],
            weights,
            info = (; loss, ∇L_norm, trajectory_references = [c.reference for c in candidates]),
            solver.rng,
        )
    end

    JointStrategy(γs)
end
