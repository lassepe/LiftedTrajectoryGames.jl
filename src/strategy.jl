"""
A potentially non-deterministic strategy that mixes over multiple continuous trajectories.
"""
Base.@kwdef struct LiftedTrajectoryStrategy{TC,TW,TI,TR} <: AbstractStrategy
    "Player index"
    player_i::Int
    "A vector of action candidates in continuous domain."
    trajectory_candidates::Vector{TC}
    "A collection of weights associated with each candidate aciton to mix over these candidates."
    weights::TW
    "A dict-like object with additioal information about this strategy."
    info::TI
    "A random number generator to compute pseudo-random actions."
    rng::TR
    "The index of the action that has been sampled when this strategy has been querried for an \
    action the first time (needed for longer open-loop rollouts)"
    action_index::Ref{Int} = Ref(0)
end

function (strategy::LiftedTrajectoryStrategy)(state, t)
    if t == 1
        strategy.action_index[] = sample(strategy.rng, Weights(strategy.weights))
    end

    (; xs, us) = strategy.trajectory_candidates[strategy.action_index[]]

    if xs[t] != state[Block(strategy.player_i)]
        throw(ArgumentError("""
                            This strategy is only valid for states states on its trajectory \
                            but has been called for an off trajectory state instead which will \
                            likely not produce meaningful results.
                            """))
    end

    PrecomputedAction(xs[t], us[t], xs[t + 1])
end

struct PrecomputedAction{TS,TC,TN}
    # TODO: Fix before merging anywhere! Abusing `reference_state` to store control input
    reference_state::TS
    reference_control::TC
    next_substate::TN
end

function TrajectoryGamesBase.join_actions(actions::AbstractVector{<:PrecomputedAction})
    joint_reference_state = mortar([a.reference_state for a in actions])
    joint_reference_control = mortar([a.reference_control for a in actions])

    joint_next_state = mortar([a.next_substate for a in actions])
    PrecomputedAction(joint_reference_state, joint_reference_control, joint_next_state)
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
