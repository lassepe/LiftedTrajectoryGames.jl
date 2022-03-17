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
    rng = Random.MersenneTwister(1),
    initial_parameters = (:random, :random),
    n_actions = (2, 2),
    reference_generator_constructors = (NNActionGenerator, NNActionGenerator),
    learning_rates = (0.05, 0.05),
    trajectory_parameterizations = (
        InputReferenceParameterization(; α = 3, params_abs_max = 10),
        InputReferenceParameterization(; α = 3, params_abs_max = 10),
    ),
    coupling_constraints_handler = LangrangianCouplingConstraintHandler(100),
    trajectory_solver = QPSolver(),
    dual_regularization_weights = (1e-4, 1e-4),
    finite_game_solver = FiniteGames.LemkeHowsonGameSolver(),
    enable_learning = (true, true),
    trajectory_caches = (nothing, nothing),
    gradient_clipping_threshold = nothing,
    execution_policy = SequentialExecutionPolicy(),
)
    num_players(game) == 2 ||
        error("Currently, only 2-player problems are supported by this solver.")

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
    candidates_per_player = map_threadable((1, 2), solver.execution_policy) do ii
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

function forward_pass(;
    solver,
    game,
    initial_state,
    min_action_probability,
    enable_caching_per_player,
)
    candidates_per_player =
        generate_trajectory_candidates(solver, initial_state, enable_caching_per_player;)

    trajectory_pairings = Zygote.ignore() do
        Iterators.product(eachindex(candidates_per_player[1]), eachindex(candidates_per_player[2])) |> collect
    end

    #    if !isnothing(solver.dual_generators)
    #        ## primal dual stuff
    #        shared_constraints = mapreduce(vcat, trajectory_pairings) do (ii, jj)
    #            τ_1 = candidates_per_player[1][ii].trajectory
    #            τ_2 = candidates_per_player[2][jj].trajectory
    #        end
    #
    #        # TODO: the call to `only` shoud be elminated by having a more appropriate output format for
    #        # the dual generator
    #        duals_per_player = [solver.dual_generators[ii](initial_state) |> only for ii in (1, 2)]
    #
    #        constraint_penalty_per_player =
    #            [-sum(duals_per_player[ii] .* shared_constraints) for ii in (1, 2)]
    #    else
    #        constraint_penalty_per_player = [0 for _ in (1, 2)]
    #    end

    # f
    # Evaluate the functions on all joint trajectories in the cost tensor
    cost_tensor = map_threadable(trajectory_pairings, solver.execution_policy) do (i1, i2)
        t1 = candidates_per_player[1][i1].trajectory
        t2 = candidates_per_player[2][i2].trajectory

        xs = map(t1.xs, t2.xs) do x1, x2
            mortar([x1, x2])
        end
        us = map(t1.us, t2.us) do u1, u2
            mortar([u1, u2])
        end

        solver.coupling_constraints_handler(game, xs, us)
    end

    # transpose matrix of tuples to tuple of matrices
    costs_per_player = map((1, 2)) do player_i
        map(cost_tensor) do pairing
            pairing[player_i]
        end
    end

    # BMG
    mixing_strategies = let
        sol = FiniteGames.solve_mixed_nash(
            solver.finite_game_solver,
            costs_per_player[1],
            costs_per_player[2];
            ϵ = min_action_probability,
        )
        (; sol.x, sol.y)
    end

    # L
    game_value_per_player = FiniteGames.game_cost(
        mixing_strategies.x,
        mixing_strategies.y,
        costs_per_player[1],
        costs_per_player[2],
    )
    dual_regularizations =
        (
            sum(sum(huber.(c.trajectory.λs)) for c in candidates_per_player[1]),
            sum(sum(huber.(c.trajectory.λs)) for c in candidates_per_player[2]),
        ) ./ solver.planning_horizon

    loss_per_player = (
        game_value_per_player.V1 + solver.dual_regularization_weights[1] * dual_regularizations[1],
        game_value_per_player.V2 + solver.dual_regularization_weights[2] * dual_regularizations[2],
    )

    info = (; game_value_per_player, mixing_strategies, candidates_per_player)

    (; loss_per_player, info)
end

function cost_gradients(back, solver, ::GeneralSumCostStructure)
    if solver.enable_learning[1]
        ∇L_1 = back((; loss_per_player = (1, nothing), info = nothing)) |> copy
    else
        ∇L_1 = nothing
    end
    if solver.enable_learning[2]
        ∇L_2 = back((; loss_per_player = (nothing, 1), info = nothing))
    else
        ∇L_2 = nothing
    end

    (; ∇L_1, ∇L_2)
end

function cost_gradients(back, solver, ::ZeroSumCostStructure)
    isnothing(solver.coupling_constraints_handler) || error("Not implemented")
    ∇L_1 = back((; loss_per_player = (1, nothing), info = nothing))
    (; ∇L_1, ∇L_2 = -1 .* ∇L_1)
end

function TrajectoryGamesBase.solve_trajectory_game!(
    solver::LiftedTrajectoryGameSolver,
    game::TrajectoryGame{<:ProductDynamics},
    initial_state;
    min_action_probability = 0.05,
    enable_caching_per_player = (false, false),
    parameter_noise = 0.0,
    scale_action_gradients = true,
)
    if !isnothing(solver.enable_learning) && any(solver.enable_learning)
        trainable_params = if !isnothing(solver.coupling_constraints_handler)
            Flux.params(solver.trajectory_parameter_generators..., solver.coupling_constraints_handler)
        else
            Flux.params(solver.trajectory_parameter_generators...)
        end
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
        (; ∇L_1, ∇L_2) = cost_gradients(back, solver, game.cost.structure)
    else
        forward_pass_result = forward_pass(;
            solver,
            game,
            initial_state,
            min_action_probability,
            enable_caching_per_player,
        )
        ∇L_1 = nothing
        ∇L_2 = nothing
    end

    ∇L_per_player = (∇L_1, ∇L_2)

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

            # TODO: re-introduce shared constraint handler parameter update here
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
