#== NNActionGenerator ==#

struct NNActionGenerator{M,O,G}
    model::M
    optimizer::O
    n_actions::Int
    gradient_clipping_threshold::G
end
@functor NNActionGenerator (model,)

function NNActionGenerator(;
    input_dimension,
    parameter_dimension,
    n_actions,
    learning_rate,
    rng,
    initial_parameters,
    params_abs_max = 10.0,
    hidden_dim = 100,
    n_hidden_layers = 2,
    output_activation = tanh,
    gradient_clipping_threshold = nothing,
)
    if initial_parameters === :random
        init = (in, out) -> Flux.glorot_uniform(rng, in, out)
    elseif initial_parameters === :all_zero
        init = (in, out) -> zeros(in, out)
    else
        @assert false
    end
    model = Chain(
        Dense(input_dimension, hidden_dim, tanh; init),
        (Dense(hidden_dim, hidden_dim, tanh; init) for _ in 1:(n_hidden_layers - 1))...,
        Dense(hidden_dim, parameter_dimension * n_actions, output_activation; init),
        x -> params_abs_max .* x,
    )
    optimizer = Optimise.Descent(learning_rate)
    NNActionGenerator(model, optimizer, n_actions, gradient_clipping_threshold)
end

function (g::NNActionGenerator)(x)
    stacked_goals = g.model(x)
    reshape(stacked_goals, :, g.n_actions)
end

function preprocess_gradients!(∇, reference_generator::NNActionGenerator, θ; kwargs...)
    if !isnothing(reference_generator.gradient_clipping_threshold)
        v = maximum(θ) do p
            maximum(g -> abs(g), ∇[p])
        end

        if v > reference_generator.gradient_clipping_threshold
            for p in θ
                ∇[p] .*= reference_generator.gradient_clipping_threshold / v
            end
        end
    end

    ∇
end

#== OnlineOptimizationActionGenerator ==#

struct OnlineOptimizationActionGenerator{T<:AbstractMatrix,O}
    params::T
    optimizer::O
end

function OnlineOptimizationActionGenerator(;
    input_dimension = nothing,
    n_actions,
    parameter_dimension,
    learning_rate,
    rng,
    initial_parameters = nothing,
    # this solver does not support gradient clipping for now
    gradient_clipping_threshold::Nothing = nothing,
)
    params = if isnothing(initial_parameters)
        (rand(rng, parameter_dimension, n_actions) .- 0.5) .* (2params_abs_max)
    else
        initial_parameters
    end
    @assert length(params) == parameter_dimension * n_actions
    optimizer = ParameterSchedulers.Scheduler(
        ParameterSchedulers.Exp(; λ = learning_rate, γ = 0.995),
        Optimise.Descent(),
    )

    OnlineOptimizationActionGenerator(params, optimizer)
end
@functor OnlineOptimizationActionGenerator (params,)

function (g::OnlineOptimizationActionGenerator)(_)
    copy(g.params)
end

function preprocess_gradients!(∇, ::OnlineOptimizationActionGenerator, θ; action_gradient_scaling)
    p = only(θ)
    ∇[p] .*= action_gradient_scaling'
end

#=== shared implementations ===#

function update_parameters!(g, ∇; noise = nothing, rng = nothing, kwargs...)
    θ = Flux.params(g)
    preprocess_gradients!(∇, g, θ; kwargs...)
    Optimise.update!(g.optimizer, θ, ∇)

    if !isnothing(noise)
        for p in θ
            p .+= randn(rng, size(p)) * noise
        end
    end
    nothing
end
