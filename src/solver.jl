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
    "The solver for the high-level finite game"
    finite_game_solver::T7
    "A flag that can be set to enable/disable learning"
    enable_learning::T8
    "A vector of cached trajectories for each player"
    trajectory_caches::T9
    "An AbstractExecutionPolicy that determines whether the solve is computed in parallel or
    sequentially."
    execution_policy::T10
end

"""
Convenience constructor to drive a suitable solver directly form a given game.
"""
function LiftedTrajectoryGameSolver(
    game::TrajectoryGame{<:ProductDynamics},
    planning_horizon;
    n_players = 2,
    rng = Random.MersenneTwister(1),
    initial_parameters = [:random for _ ∈ 1:n_players],
    n_actions = 2ones(Int,n_players),
    reference_generator_constructors = [NNActionGenerator for _ ∈ 1:n_players],
    learning_rates = 0.05*ones(n_players),
    trajectory_parameterizations = [ 
        InputReferenceParameterization(; α = 3, params_abs_max = 10) for _ ∈ 1:n_players
    ],
    coupling_constraints_handler = LangrangianCouplingConstraintHandler(100),
    trajectory_solver = QPSolver(),
    dual_regularization_weights = 1e-4*ones(n_players),
    finite_game_solver = FiniteGames.TensorGameSolver(),
    enable_learning = ones(Bool, n_players),
    trajectory_caches = [nothing for _ ∈ 1:n_players],
    gradient_clipping_threshold = nothing,
    execution_policy = SequentialExecutionPolicy(),
)
    #num_players(game) == 2 ||
    #    error("Currently, only 2-player problems are supported by this solver.")

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
        finite_game_solver,
        enable_learning,
        trajectory_caches,
        execution_policy,
    )
end

function huber(x; δ = 1)
    if abs(x) > δ
        δ * (abs(x) - 0.5δ)
    else
        0.5x^2
    end
end

function generate_trajectory_candidates(solver, initial_state, enable_caching_per_player;)
    state_per_player = blocks(initial_state)
    n_players = length(state_per_player)
    candidates_per_player = map_threadable([1:n_players...], solver.execution_policy) do ii
        cache = solver.trajectory_caches[ii]
        if !isnothing(cache) && enable_caching_per_player[ii]
            cache
        else
            # π_θi
            references = solver.trajectory_parameter_generators[ii](initial_state)
            # TRAJ_i
            trajectory_generator = solver.trajectory_generators[ii]
            substate = state_per_player[ii]
            map_threadable(
                reference ->
                    (; trajectory = trajectory_generator(substate, reference), reference),
                references,
                solver.execution_policy,
            )
        end
    end
    candidates_per_player
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
    candidates_per_player =
        generate_trajectory_candidates(solver, initial_state, enable_caching_per_player;)

    trajectory_pairings = Zygote.ignore() do
        Iterators.product([eachindex(candidates_per_player[i]) for i ∈ 1:n_players]...) |> collect
    end

    # f
    # Evaluate the functions on all joint trajectories in the cost tensor
    cost_tensor = map_threadable(trajectory_pairings, solver.execution_policy) do (i)
        trajectories = [candidates_per_player[j][i[j]].trajectory for j ∈ 1:n_players]

        xs = map([t.xs for t ∈ trajectories]...) do x...
            mortar([x...])
        end

        us = map([t.us for t ∈ trajectories]...) do u...
            mortar([u...])
        end

        solver.coupling_constraints_handler(game, xs, us)
    end

    # transpose tensor of tuples to tuple of tensors 
    costs_per_player = map(1:n_players) do player_i
        map(cost_tensor) do pairing
            pairing[player_i]
        end
    end

    # BMG
    mixing_strategies = let
        sol = FiniteGames.solve_mixed_nash(
            solver.finite_game_solver,
            costs_per_player;
            ϵ = min_action_probability,
        )
        sol.x
    end

    # L
    game_value_per_player = FiniteGames.game_cost(
        mixing_strategies,
        costs_per_player,
    )
    dual_regularizations =
        [sum(sum(huber.(c.trajectory.λs)) for c in candidates_per_player[i]) for i ∈ 1:n_players]./ solver.planning_horizon

    loss_per_player = [ 
        game_value_per_player.V[i] + solver.dual_regularization_weights[i] * dual_regularizations[i] for i ∈ 1:n_players
    ] 

    info = (; game_value_per_player, mixing_strategies, candidates_per_player)

    (; loss_per_player, info)
end

function cost_gradients(back, solver, n_players, ::GeneralSumCostStructure)
    ∇L = ( begin    
        if solver.enable_learning[n]
            loss_per_player = ( i==n ? 1 : nothing for i ∈ 1:n_players)
            back((; loss_per_player, info=nothing)) |> copy
        else
            nothing
        end
    end
    for n ∈ 1:n_players
    )
end

function cost_gradients(back, solver, n_players, ::ZeroSumCostStructure)
    n_players == 2 || error("Not implemented for N>2 players")
    isnothing(solver.coupling_constraints_handler) || error("Not implemented")
    ∇L_1 = back((; loss_per_player = (1, nothing), info = nothing))
    ∇L = (∇L_1, -1 .* ∇L_2)
end

function TrajectoryGamesBase.solve_trajectory_game!(
    solver::LiftedTrajectoryGameSolver,
    game::TrajectoryGame{<:ProductDynamics},
    initial_state;
    min_action_probability = 0.0,
    enable_caching_per_player = zeros(Bool, num_players(game)),
    parameter_noise = 0.0,
    scale_action_gradients = true,
)
    n_players = num_players(game)
    if !isnothing(solver.enable_learning) && any(solver.enable_learning)
        trainable_params = Flux.params(solver.trajectory_parameter_generators...)
        forward_pass_result, back = Zygote.pullback(
            () -> forward_pass(;
                solver,
                game,
                initial_state,
                min_action_probability,
                enable_caching_per_player,
            ),
            trainable_params,
        )
        ∇L_per_player = cost_gradients(back, solver, n_players, game.cost.structure)
    else
        forward_pass_result = forward_pass(;
            solver,
            game,
            initial_state,
            min_action_probability,
            enable_caching_per_player,
        )
        ∇L_per_player = (nothing for n ∈ 1:n_players)
    end

    (; loss_per_player, info) = forward_pass_result

    # Store computed trajectories in caches if caching is enabled
    if !(eltype(solver.trajectory_caches) <: Nothing)
        for ii in eachindex(info.candidates_per_player)
            if enable_caching_per_player[ii]
                solver.trajectory_caches[ii] = info.candidates_per_player[ii]
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
