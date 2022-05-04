__precompile__(false)

module LiftedTrajectoryGames

using DifferentiableTrajectoryGenerators:
    DifferentiableTrajectoryGenerator,
    ParametricOptimizationProblem,
    InputReferenceParameterization,
    GoalReferenceParameterization,
    QPSolver,
    parameter_dimension
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
using Flux: Flux, Chain, Dense, Optimise, leakyrelu, @functor
using Makie: Makie
using StatsBase: Weights, sample
using Random: Random
using Zygote: Zygote
using LinearAlgebra: norm
using ParameterSchedulers: ParameterSchedulers
using ForwardDiff: ForwardDiff
using TensorGames: TensorGames

# multi-threading extensions
using ThreadsX: ThreadsX
using ChainRulesCore: ChainRulesCore
include("threadsx_chainrules_piracy.jl")
include("execution_policy.jl")

include("coupling_constraint_handlers.jl")
include("execution_policy.jl")
include("solver.jl")
include("statevalue_predictors.jl")
include("strategy.jl")
include("trajectory_reference_generators.jl")
include("visualization.jl")

end
