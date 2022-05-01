struct LangrangianCouplingConstraintHandler
    violation_penalty::Float64
end

function (constraint_handler::LangrangianCouplingConstraintHandler)(
    game,
    xs,
    us,
    context_state,
)
    costs = game.cost(xs, us, context_state)

    if isnothing(game.coupling_constraints)
        costs
    else
        constraint_penalties = [
            sum(coupling_constraints_per_player(xs, us)) do g
                if g >= 0
                    # the constraint is already satsified, no penalty
                    zero(g)
                else
                    -g * constraint_handler.violation_penalty
                end
            end for coupling_constraints_per_player in game.coupling_constraints
        ]

        # lagrangian approximation to enforce coupling constraints
        costs + constraint_penalties
    end
end
