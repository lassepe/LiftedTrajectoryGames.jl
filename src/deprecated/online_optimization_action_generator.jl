struct OnlineOptimizationActionGenerator{T<:AbstractMatrix,O}
    params::T
    params_abs_max::Float64
    optimizer::O
end

function OnlineOptimizationActionGenerator(;
    n_actions,
    n_params,
    params_abs_max,
    learning_rate,
    rng,
)
    params = (rand(rng, n_params, n_actions) .- 0.5) .* params_abs_max
    optimizer = Optimise.Descent(learning_rate)

    OnlineOptimizationActionGenerator(params, params_abs_max, optimizer)
end

function (g::OnlineOptimizationActionGenerator)(_)
    [p for p in eachcol(g.params)]
end

@functor OnlineOptimizationActionGenerator (params,)

function update_parameters!(g::OnlineOptimizationActionGenerator, ∇)
    p = params(g)
    Optimise.update!(g.optimizer, p, ∇)
    clamp!(g.params, -g.params_abs_max, g.params_abs_max)
    nothing
end
