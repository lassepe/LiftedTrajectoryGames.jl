Base.@kwdef struct LiftedTrajectoryGameSolver{TA,TT,TH,TF,TS,TR,TL,TC}
    "A collection of action generators, one for each player in the game."
    trajectory_parameter_generators::TA
    "A acollection of trajectory generators, one for each player in the game"
    trajectory_generators::TT
    "The number of time steps to plan into the future."
    planning_horizon::TH
    "The solver for the high-level finite game"
    finite_game_solver::TF = FiniteGames.LemkeHowsonGameSolver()
    "A value network to predict the game value for each player"
    statevalue_predictor::TS = nothing
    "A random number generator to generate non-deterministic strategies."
    rng::TR = Random.MersenneTwister(1)
    "A flag that can be set to enable/disable learning"
    enable_learning::TL = (true, true)
    "A vector of cached trajectories for each player"
    trajectory_caches::TC = (nothing, nothing)
end

"""
Convenience constructor to drive a suitable solver directly form a given game.
"""
function LiftedTrajectoryGameSolver(
    game::TrajectoryGame{<:ProductDynamics,<:ZeroSumTrajectoryGameCost},
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

    signed_learning_rates = map(*, learning_rates, (1.0, -1.0))

    trajectory_parameter_generators = map(
        reference_generator_constructors,
        trajectory_generators,
        n_actions,
        initial_parameters,
        signed_learning_rates,
    ) do constructor,
    trajectory_generator,
    n_player_actions,
    initial_player_parameters,
    signed_learning_rate
        constructor(;
            state_dim = state_dim(game.dynamics),
            n_params = param_dim(trajectory_generator),
            n_actions = n_player_actions,
            trajectory_generator.problem.parameterization.params_abs_max,
            learning_rate = signed_learning_rate,
            rng,
            initial_parameters = initial_player_parameters,
        )
    end

    LiftedTrajectoryGameSolver(;
        trajectory_parameter_generators,
        trajectory_generators,
        planning_horizon,
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

function forward_pass(;
    solver,
    game,
    initial_state,
    dual_regularization_weight,
    min_action_probability,
    enable_caching,
)
    player_references = map(gen -> gen(initial_state), solver.trajectory_parameter_generators)

    player_trajectory_candidates = map(
        blocks(initial_state),
        enable_caching,
        solver.trajectory_caches,
        player_references,
        solver.trajectory_generators,
    ) do substate, enable_caching, cache, refs, trajectory_generator
        if !isnothing(cache) && enable_caching
            cache
        else
            [trajectory_generator(substate, ref) for ref in refs]
        end
    end

    cost_tensor = map(
        Iterators.product(
            eachindex(player_trajectory_candidates[1]),
            eachindex(player_trajectory_candidates[2]),
        ),
    ) do (i1, i2)
        t1 = player_trajectory_candidates[1][i1]
        t2 = player_trajectory_candidates[2][i2]

        xs = map(t1.xs, t2.xs) do x1, x2
            mortar([x1, x2])
        end
        us = map(t1.us, t2.us) do u1, u2
            mortar([u1, u2])
        end
        game.cost(xs, us)
    end

    mixing_strategies = let
        sol = FiniteGames.solve_mixed_nash(
            solver.finite_game_solver,
            cost_tensor;
            ϵ = min_action_probability,
        )
        (; x=sol.x[1], y=sol.x[2])
    end

    Vs = FiniteGames.game_cost(mixing_strategies.x, mixing_strategies.y, cost_tensor).V
    dual_regularization =
        (
            sum(sum(huber.(t.λs)) for t in player_trajectory_candidates[1]) -
            sum(sum(huber.(t.λs)) for t in player_trajectory_candidates[2])
        ) / solver.planning_horizon

    loss = Vs[1] + dual_regularization_weight * dual_regularization

    (; loss, Vs, mixing_strategies, player_trajectory_candidates)
end

# TODO: Re-introduce state-value learning
function TrajectoryGamesBase.solve_trajectory_game!(
    solver::LiftedTrajectoryGameSolver,
    game::TrajectoryGame{<:ProductDynamics,<:ZeroSumTrajectoryGameCost},
    initial_state;
    dual_regularization_weight = 1e-4,
    min_action_probability = 0.05,
    enable_caching = (false, false),
)
    # TODO: make this a parameter
    parameter_noise = 0.0
    scale_action_gradients = true

    ∇V1 = if !isnothing(solver.enable_learning) && any(solver.enable_learning)
        res, back = Zygote.pullback(
            () -> forward_pass(;
                solver,
                game,
                initial_state,
                dual_regularization_weight,
                min_action_probability,
                enable_caching,
            ),
            Flux.params(solver.trajectory_parameter_generators...),
        )
        back((;
            loss = 1,
            Vs = nothing,
            mixing_strategies = nothing,
            player_trajectory_candidates = nothing,
        ))
    else
        res = forward_pass(;
            solver,
            game,
            initial_state,
            dual_regularization_weight,
            min_action_probability,
            enable_caching,
        )
        nothing
    end

    (; Vs, mixing_strategies, player_trajectory_candidates) = res

    if !(eltype(solver.trajectory_caches) <: Nothing)
        for ii in eachindex(player_trajectory_candidates)
            if enable_caching[ii]
                solver.trajectory_caches[ii] = player_trajectory_candidates[ii]
            end
        end
    end

    γs = map(
        Iterators.countfrom(),
        mixing_strategies,
        Vs,
        player_trajectory_candidates,
    ) do player_i, weights, V, trajectory_candidates
        info = (;
            V,
            # TODO: maybe allow to disable
            ∇_norm = if isnothing(∇V1)
                0.0
            else
                sum(
                    norm(something(∇V1[p], 0.0)) for
                    p in Flux.params(solver.trajectory_parameter_generators...)
                )
            end,
        )
        LiftedTrajectoryStrategy(; player_i, trajectory_candidates, weights, info, solver.rng)
    end

    if !isnothing(solver.enable_learning)
        for (parameter_generator, weights, enable_player_learning) in
            zip(solver.trajectory_parameter_generators, mixing_strategies, solver.enable_learning)
            if !enable_player_learning
                continue
            end
            action_gradient_scaling = scale_action_gradients ? 1 ./ weights : ones(size(weights))
            update_parameters!(
                parameter_generator,
                ∇V1;
                noise = parameter_noise,
                solver.rng,
                action_gradient_scaling,
            )
        end
    end

    JointStrategy(γs)
end
