"""
A potentially non-deterministic strategy that mixes over multiple continuous trajectories.
"""
struct LiftedTrajectoryStrategy{TC,TW,TX,TI,TR} <: AbstractStrategy
    "A vector of action candidates in continuous domain."
    trajectory_candidates::Vector{TC}
    "A collection of weights associated with each candidate aciton to mix over these candidates."
    weights::TW
    "The reference state for which this strategy has been computed"
    reference_state::TX
    "A dict-like object with additioal information about this strategy."
    info::TI
    "A random number generator to compute pseudo-random actions."
    rng::TR
end

function (strategy::LiftedTrajectoryStrategy)(state, t = nothing)
    # TODO: get turnlength from somewhere else
    turn_length = 5

    if strategy.reference_state != state
        throw(ArgumentError("""
                            This strategy is only valid for states $(strategy.reference_state) \
                            but has been called for $state instead which will likely not \
                            produce meaningful results.
                            """))
    end

    action_index = sample(strategy.rng, Weights(strategy.weights))
    trajectory = strategy.trajectory_candidates[action_index]
    next_substate = trajectory[turn_length]

    PrecomputedAction(strategy.reference_state, next_substate)
end

struct PrecomputedAction{TR,TN}
    reference_state::TR
    next_substate::TN
end

function TrajectoryGamesBase.join_actions(actions::AbstractVector{<:PrecomputedAction})
    reference_state = only(unique(a.reference_state for a in actions))
    joint_next_state = mortar([a.next_substate for a in actions])
    PrecomputedAction(reference_state, joint_next_state)
end

function (dynamics::AbstractDynamics)(state, action::PrecomputedAction, t = nothing)
    if action.reference_state != state
        throw(
            ArgumentError("""
                          This precomputed action is only valid for states \
                          $(action.reference_state) but has been called for $state instead which \
                          will likely not produce meaningful results.
                          """),
        )
    end
    action.next_substate
end
