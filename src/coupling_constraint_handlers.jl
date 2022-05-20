struct LagrangianCouplingConstraintHandler
    violation_penalty::Float64
end

function (constraint_handler::LagrangianCouplingConstraintHandler)(game, xs, us, context_state)
    constraint_penalties = map(game.coupling_constraints) do coupling_constraints_per_player
        sum(coupling_constraints_per_player(xs, us)) do g
            if g >= 0
                # the constraint is already satsified, no penalty
                zero(g)
            else
                -g * constraint_handler.violation_penalty
            end
        end
    end

    # lagrangian approximation to enforce coupling constraints
    constraint_penalties
end
