#== NNActionGenerator ==#

struct NNActionGenerator{M,O}
    model::M
    optimizer::O
    n_actions::Int
end
@functor NNActionGenerator (model,)

function NNActionGenerator(;
    state_dim,
    hidden_dim,
    n_hidden_layers,
    n_params,
    params_abs_max,
    n_actions,
    learning_rate,
    rng,
    initial_parameter_population::Nothing,
)
    init(in, out) = Flux.glorot_uniform(rng, in, out)

    model = Chain(
        Dense(state_dim, hidden_dim, tanh; init),
        (Dense(hidden_dim, hidden_dim, tanh; init) for _ in 1:(n_hidden_layers - 1))...,
        Dense(hidden_dim, n_params * n_actions, tanh; init),
        x -> params_abs_max .* x,
    )

    optimizer = Optimise.Descent(learning_rate)

    NNActionGenerator(model, optimizer, n_actions)
end

function (g::NNActionGenerator)(states)
    x = reduce(vcat, states)
    stacked_goals = g.model(x)
    collect(eachcol(reshape(stacked_goals, :, g.n_actions)))
end

function update_parameters!(g, ∇; noise = nothing, rng = nothing, action_gradient_scaling)
    θ = Flux.params(g)
    p = only(θ)
    ∇[p] .*= action_gradient_scaling'
    Optimise.update!(g.optimizer, θ, ∇)

    if !isnothing(noise)
        for p in θ
            p .+= randn(rng, size(p)) * noise
        end
    end
    nothing
end

#== OnlineOptimizationActionGenerator ==#

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
    initial_parameter_population = nothing,
)
    params = if isnothing(initial_parameter_population)
        (rand(rng, n_params, n_actions) .- 0.5) .* (2params_abs_max)
    else
        reduce(hcat, sample(rng, initial_parameter_population, n_actions; replace = false))
    end
    @assert length(params) == n_params * n_actions
    optimizer = ParameterSchedulers.Scheduler(
        ParameterSchedulers.Exp(; λ = learning_rate, γ = 0.995),
        Optimise.Descent(),
    )

    OnlineOptimizationActionGenerator(params, optimizer)
end
@functor OnlineOptimizationActionGenerator (params,)

function (g::OnlineOptimizationActionGenerator)(_)
    collect(eachcol(g.params))
end
