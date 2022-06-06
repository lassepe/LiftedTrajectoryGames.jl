__precompile__(false)

module LiftedTrajectoryGames

using DifferentiableTrajectoryOptimization:
    DifferentiableTrajectoryOptimization as Dito,
    Optimizer,
    ParametricTrajectoryOptimizationProblem,
    QPSolver,
    get_constraints_from_box_bounds,
    is_thread_safe,
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
    control_dim,
    state_bounds,
    control_bounds,
    num_players,
    join_actions,
    Block,
    blocks,
    mortar,
    get_constraints,
    cost_structure_trait
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

include("dito_parameterizations.jl")
include("coupling_constraint_handlers.jl")
include("execution_policy.jl")
include("solver.jl")
include("statevalue_predictors.jl")
include("strategy.jl")
include("trajectory_reference_generators.jl")
include("visualization.jl")

end
