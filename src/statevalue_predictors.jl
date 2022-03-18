struct NNValuePredictor{M,O}
    model::M
    optimizer::O
end

@functor NNValuePredictor (model,)

function NNValuePredictor(; game, learning_rate, rng, n_hidden_layers = 2, hidden_dim = 100)
    init(in, out) = Flux.glorot_uniform(rng, in, out)
    model = Chain(
        Dense(state_dim(game.dynamics), hidden_dim, sin; init),
        (Dense(hidden_dim, hidden_dim, sin; init) for _ in 1:(n_hidden_layers - 1))...,
        Dense(hidden_dim, num_players(game); init),
    )

    optimizer = Optimise.Descent(learning_rate)

    NNValuePredictor(model, optimizer)
end

function (m::NNValuePredictor)(states)
    s = reduce(vcat, states)
    m.model(s)
end

function fit_value_predictor!(state_value_predictor, state_value_replay_buffer, n_value_epochs)
    for _ in 1:n_value_epochs
        ∇L = Zygote.gradient(params(state_value_predictor)) do
            sum(state_value_replay_buffer) do d
                states = [p.position for p in d.players]
                (d.V1_cl - state_value_predictor(states))^2
            end / length(state_value_replay_buffer)
        end
        update_parameters!(state_value_predictor, ∇L)
    end
end
