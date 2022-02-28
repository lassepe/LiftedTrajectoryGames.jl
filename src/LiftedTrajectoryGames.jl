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
    state_dim,
    num_players,
    join_actions,
    Block,
    blocks,
    mortar
using Flux: Flux, Chain, Dense, Optimise, @functor
using Makie: Makie, @L_str
using StatsBase: Weights, sample
using Random: Random
using Zygote: Zygote
using LinearAlgebra: norm
using ParameterSchedulers: ParameterSchedulers

include("trajectory_parameter_generators.jl")
include("statevalue_predictors.jl")
include("strategy.jl")
include("solver.jl")
include("visualization.jl")

end
