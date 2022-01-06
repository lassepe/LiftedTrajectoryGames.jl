"""
A potentially non-deterministic strategy that mixes over multiple continuous trajectories.
"""
struct LiftedTrajectoryStrategy{TC,TW,TX,TI,TR}
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

function (dynamics::ProductDynamics)(
    state,
    strategies::Vector{<:LiftedTrajectoryStrategy},
    t = nothing,
)
    # TODO: get turnlength from somewhere else
    turn_length = 5
    map(strategies) do γ
        if γ.reference_state != state
            throw(ArgumentError("""
                                This strategy is only valid for states $(γ.reference_state) \
                                but has been called for $state instead which will likely not \
                                produce meaningful results.
                                """))
        end

        action_index = sample(γ.rng, Weights(γ.weights))
        trajectory = γ.trajectory_candidates[action_index]
        trajectory[:, turn_length]
    end |> mortar
end
