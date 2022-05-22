"""
Computes the number of scalar parameters that this parameterization consists of when applied to
a problem of horizon `horizon` and stage-wise dimensions of states, `state_dim`, and controls
`control_dim`.
"""
_parameter_dimension(parameterization; horizon::Integer, state_dim::Integer, control_dim::Integer)

#== InputReferenceParameterization ==#

Base.@kwdef struct InputReferenceParameterization
    α::Float64
end

function (parameterization::InputReferenceParameterization)(xs, us, params)
    horizon = length(us)
    ps = reshape(params, :, horizon) |> eachcol
    sum(zip(xs, us, ps)) do (x, u, param)
        sum(0.5 .* parameterization.α .* u .^ 2 .- u .* param)
    end
end

function _parameter_dimension(::InputReferenceParameterization; horizon, state_dim, control_dim)
    horizon * control_dim
end

#== GoalReferenceParameterization==#

Base.@kwdef struct GoalReferenceParameterization
    α::Float64
end

function (parameterization::GoalReferenceParameterization)(xs, us, params)
    sum(zip(xs, us)) do (x, u)
        sum((x[1:2] - params) .^ 2) + parameterization.α * sum(u .^ 2)
    end
end

function _parameter_dimension(::GoalReferenceParameterization; horizon, state_dim, control_dim)
    2
end
