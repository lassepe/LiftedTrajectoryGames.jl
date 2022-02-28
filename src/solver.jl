Base.@kwdef struct LiftedTrajectoryGameSolver{TA,TT,TH,TF,TR,TL,TC,TD}
    "A collection of action generators, one for each player in the game."
    trajectory_parameter_generators::TA
    "A acollection of trajectory generators, one for each player in the game"
    trajectory_generators::TT
    "The number of time steps to plan into the future."
    planning_horizon::TH
    "A random number generator to generate non-deterministic strategies."
    rng::TR
    "How much affect the dual regularization has on the costs"
    dual_regularization_weights::TD = (1e-4, 1e-4)
    "The solver for the high-level finite game"
    finite_game_solver::TF = FiniteGames.LemkeHowsonGameSolver()
    "A flag that can be set to enable/disable learning"
    enable_learning::TL = (true, true)
    "A vector of cached trajectories for each player"
    trajectory_caches::TC = (nothing, nothing)
end

"""
Convenience constructor to drive a suitable solver directly form a given game.
"""
function LiftedTrajectoryGameSolver(
    game::TrajectoryGame{<:ProductDynamics,<:AbstractTrajectoryGameCost},
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
    trajectory_solver = QPSolver(),
    kwargs...,
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
        )
    end

    LiftedTrajectoryGameSolver(;
        trajectory_parameter_generators,
        trajectory_generators,
        planning_horizon,
        rng,
        kwargs...,
    )
end

function huber(x; δ = 1)
    if abs(x) > δ
        δ * (abs(x) - 0.5δ)
    else
        0.5x^2
    end
end

function generate_trajectory_candidates(
    initial_state,
    references_per_player,
    trajectory_generators,
    enable_caching_per_player,
    trajectory_caches,
)
    state_per_player = blocks(initial_state)
    candidates_per_player = map(
        state_per_player,
        references_per_player,
        trajectory_generators,
        enable_caching_per_player,
        trajectory_caches,
    ) do state, references, trajectory_generator, enable_caching, cache
        if enable_caching && !isnothing(cache)
            cache
        else
            [trajectory_generator(state, reference) for reference in references]
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
    # π_θ
    trajectory_references_per_player =
        map(generator -> generator(initial_state), solver.trajectory_parameter_generators)
    # TRAJ
    trajectory_candidates_per_player = generate_trajectory_candidates(
        initial_state,
        trajectory_references_per_player,
        solver.trajectory_generators,
        enable_caching_per_player,
        solver.trajectory_caches,
    )

    trajectory_pairings = Iterators.product(
        eachindex(trajectory_candidates_per_player[1]),
        eachindex(trajectory_candidates_per_player[2]),
    )
    # f
    costs_per_trajectory_pairing = map(trajectory_pairings) do (i1, i2)
        t1 = trajectory_candidates_per_player[1][i1]
        t2 = trajectory_candidates_per_player[2][i2]

        xs = map(t1.xs, t2.xs) do x1, x2
            mortar([x1, x2])
        end
        us = map(t1.us, t2.us) do u1, u2
            mortar([u1, u2])
        end
        game.cost(xs, us)
    end

    # transpose matrix of tuples to tuple of matrices
    costs_per_player = map((1, 2)) do player_i
        map(costs_per_trajectory_pairing) do pairing
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
            sum(sum(huber.(t.λs)) for t in trajectory_candidates_per_player[1]),
            sum(sum(huber.(t.λs)) for t in trajectory_candidates_per_player[2]),
        ) ./ solver.planning_horizon

    loss_per_player = (
        game_value_per_player.V1 + solver.dual_regularization_weights[1] * dual_regularizations[1],
        game_value_per_player.V2 + solver.dual_regularization_weights[2] * dual_regularizations[2],
    )

    info = (;
        game_value_per_player,
        mixing_strategies,
        trajectory_candidates_per_player,
        trajectory_references_per_player,
    )

    (; loss_per_player, info)
end

function TrajectoryGamesBase.solve_trajectory_game!(
    solver::LiftedTrajectoryGameSolver,
    game::TrajectoryGame{<:ProductDynamics,<:AbstractTrajectoryGameCost},
    initial_state;
    min_action_probability = 0.05,
    enable_caching_per_player = (false, false),
    parameter_noise = 0.0,
    scale_action_gradients = true,
)
    if !isnothing(solver.enable_learning) && any(solver.enable_learning)
        forward_pass_result, back = Zygote.pullback(
            () -> forward_pass(;
                solver,
                game,
                initial_state,
                min_action_probability,
                enable_caching_per_player,
            ),
            Flux.params(solver.trajectory_parameter_generators...),
        )
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
        for ii in eachindex(info.trajectory_candidates_per_player)
            if enable_caching_per_player[ii]
                solver.trajectory_caches[ii] = info.trajectory_candidates_per_player[ii]
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
        info.trajectory_candidates_per_player,
        info.trajectory_references_per_player,
        ∇L_per_player,
    ) do player_i, weights, loss, trajectory_candidates, trajectory_references, ∇L
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
            trajectory_candidates,
            weights,
            info = (; loss, ∇L_norm, trajectory_references),
            solver.rng,
        )
    end

    JointStrategy(γs)
end
