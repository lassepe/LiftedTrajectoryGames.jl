# custom split layer
struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)
@functor Split
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

struct NeuralStateValuePredictor{T1,T2,T3}
    model::T1
    optimizer::T2
    replay_buffer::T3
    turn_length::Int
    batch_size::Int
    n_epochs_per_update::Int
end

@functor NeuralStateValuePredictor (model,)

function NeuralStateValuePredictor(;
    game,
    learning_rate,
    rng,
    turn_length,
    replay_buffer = NamedTuple[],
    output_scaling = 1,
    n_hidden_layers = 4,
    hidden_dim = 100,
    activation = leakyrelu,
    batch_size = 50,
    n_epochs_per_update = 10,
)
    init(in, out) = Flux.glorot_uniform(rng, in, out)

    model = let
        legs = map(1:num_players(game)) do ii
            Chain(
                Dense(state_dim(game.dynamics), hidden_dim, activation; init),
                (
                    Dense(hidden_dim, hidden_dim, activation; init) for
                    _ in 1:(n_hidden_layers - 1)
                )...,
                Dense(hidden_dim, 1; init),
                x -> output_scaling * x,
                only,
            )
        end
        Split(legs)
    end

    optimizer = Optimise.Descent(learning_rate)

    NeuralStateValuePredictor(
        model,
        optimizer,
        replay_buffer,
        turn_length,
        batch_size,
        n_epochs_per_update,
    )
end

function (p::NeuralStateValuePredictor)(state)
    joint_state = reduce(vcat, state)
    p.model(joint_state)
end

function preprocess_gradients!(∇, state_value_predictor::NeuralStateValuePredictor, θ; kwargs...)
    ∇
end

function fit_value_predictor!(state_value_predictor::NeuralStateValuePredictor)
    @assert length(state_value_predictor.replay_buffer) > 0

    for _ in 1:(state_value_predictor.n_epochs_per_update)
        θ = Flux.params(state_value_predictor)
        ∇L = Zygote.gradient(θ) do
            sum(state_value_predictor.replay_buffer) do d
                sum(v -> v^2, d.value_target_per_player - state_value_predictor(d.state))
            end / length(state_value_predictor.replay_buffer)
        end

        # TODO: we shouldn't have to pass the annoying `action_gradient_scaling` here
        update_parameters!(state_value_predictor, ∇L; action_gradient_scaling = 1)
    end
end
