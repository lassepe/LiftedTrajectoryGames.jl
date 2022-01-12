struct OnlineOptimizationActionGenerator{T<:AbstractMatrix,O}
    params::T
    optimizer::O
end

function OnlineOptimizationActionGenerator(;
    n_actions,
    n_params,
    params_abs_max,
    learning_rate,
    rng,
)
    params = (rand(rng, n_params, n_actions) .- 0.5) .* (2params_abs_max)
    optimizer = Optimise.Descent(learning_rate)

    OnlineOptimizationActionGenerator(params, params_abs_max, optimizer)
end

function (g::OnlineOptimizationActionGenerator)(_)
    collect(eachcol(g.params))
end

@functor OnlineOptimizationActionGenerator (params,)

function update_parameters!(g::OnlineOptimizationActionGenerator, ∇)
    p = Flux.params(g)
    Optimise.update!(g.optimizer, p, ∇)
end
