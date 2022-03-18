struct NNValuePredictor{M,O}
    model::M
    optimizer::O
end

@functor NNValuePredictor (model,)

function NNValuePredictor(; game, learning_rate, rng, n_hidden_layers = 4, hidden_dim = 100)
    init(in, out) = Flux.glorot_uniform(rng, in, out)
    model = Chain(
        Dense(state_dim(game.dynamics), hidden_dim, tanh; init),
        (Dense(hidden_dim, hidden_dim, tanh; init) for _ in 1:(n_hidden_layers - 1))...,
        Dense(hidden_dim, num_players(game); init),
    )

    optimizer = Optimise.Descent(learning_rate)

    NNValuePredictor(model, optimizer)
end

function (m::NNValuePredictor)(state)
    v = m.model(reduce(vcat, state))
    Zygote.ignore() do
        @infiltrate any(x -> isnan(x) || isinf(x), v)
    end
    v
end

function preprocess_gradients!(∇, state_value_predictor::NNValuePredictor, θ; kwargs...)
    for p in θ
        @assert !any(x -> isnan(x) || isinf(x), p)
        @assert !any(x -> isnan(x) || isinf(x), ∇[p])
    end
    ∇
end

function fit_value_predictor!(
    state_value_predictor::NNValuePredictor,
    replay_buffer,
    n_value_epochs,
)
    @assert length(replay_buffer) > 0

    for _ in 1:n_value_epochs
        θ = Flux.params(state_value_predictor)
        ∇L = Zygote.gradient(θ) do
            sum(replay_buffer) do d
                sum(v -> v^2, d.value_target_per_player - state_value_predictor(d.state))
            end / length(replay_buffer)
        end

        @assert !any(map(θ) do p
            dp = ∇L[p]
            any(x -> isnan(x) || isinf(x), p)
            any(x -> isnan(x) || isinf(x), dp)
        end)

        # TODO: we shouldn't have to pass the annoying `action_gradient_scaling` here
        update_parameters!(state_value_predictor, ∇L; action_gradient_scaling = 1)
    end
end
