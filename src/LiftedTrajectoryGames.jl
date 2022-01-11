module LiftedTrajectoryGames

using FiniteGames: FiniteGames
using DifferentiableTrajectoryGenerators:
    DifferentiableTrajectoryGenerators,
    DifferentiableTrajectoryGenerator,
    ParametricOptimizationProblem,
    InputReferenceParameterization,
    evaluate_trajectories,
    QPSolver,
    param_dim
using TrajectoryGamesBase:
    TrajectoryGamesBase,
    TrajectoryGame,
    AbstractDynamics,
    AbstractStrategy,
    JointStrategy,
    ZeroSumCostStructure,
    ProductDynamics,
    state_dim,
    num_players,
    join_actions,
    blocks,
    mortar
using Flux: Flux, Chain, Dense, Optimise, @functor
using Makie: Makie, @L_str
using StatsBase: Weights, sample
using Random: Random
using Zygote: Zygote

include("trajectory_parameter_generators.jl")
include("statevalue_predictors.jl")
include("strategy.jl")
include("solver.jl")
include("visualization.jl")

end
