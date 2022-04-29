struct LangrangianCouplingConstraintHandler
    violation_penalty::Float64
end

function (constraint_handler::LangrangianCouplingConstraintHandler)(
    game,
    xs,
    us,
    context_state,
)
    if isnothing(game.coupling_constraints)
        constraint_penalty = 0
    else
        constraint_penalty = sum(game.coupling_constraints(xs, us)) do g
            if g >= 0
                # the constraint is already satsified, no penalty
                zero(g)
            else
                -g * constraint_handler.violation_penalty
            end
        end
    end

    # lagrangian approximation to enforce coupling constraints
    game.cost(xs, us, context_state) .+ constraint_penalty
end
