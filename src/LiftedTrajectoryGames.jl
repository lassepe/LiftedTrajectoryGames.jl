__precompile__(false)

module LiftedTrajectoryGames

using FiniteGames: FiniteGames
using DifferentiableTrajectoryGenerators:
    DifferentiableTrajectoryGenerator,
    ParametricOptimizationProblem,
    InputReferenceParameterization,
    GoalReferenceParameterization,
    QPSolver,
    param_dim
using TrajectoryGamesBase:
    TrajectoryGamesBase,
    TrajectoryGame,
    AbstractDynamics,
    AbstractStrategy,
    JointStrategy,
    ProductDynamics,
    GeneralSumCostStructure,
    ZeroSumCostStructure,
    state_dim,
    num_players,
    join_actions,
    Block,
    blocks,
    mortar
using Flux: Flux, Chain, Dense, Optimise, @functor
using Makie: Makie
using StatsBase: Weights, sample
using Random: Random
using Zygote: Zygote
using LinearAlgebra: norm
using ParameterSchedulers: ParameterSchedulers
using ThreadsX: ThreadsX
using ThreadsXChainRules: ThreadsXChainRules

include("execution_policy.jl")
include("trajectory_parameter_generators.jl")
include("statevalue_predictors.jl")
include("strategy.jl")
include("coupling_constraint_handlers.jl")
include("solver.jl")
include("visualization.jl")

end
