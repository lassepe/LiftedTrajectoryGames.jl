# custom split layer
struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)
@functor Split
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

struct NNValuePredictor{M,O}
    model::M
    optimizer::O
end

@functor NNValuePredictor (model,)

function NNValuePredictor(;
    game,
    learning_rate,
    rng,
    output_scaling = 1,
    n_hidden_layers = 4,
    hidden_dim = 100,
    activation = leakyrelu,
)
    init(in, out) = Flux.glorot_uniform(rng, in, out)

    # TODO: use Flux.Parallel for a decoupled architecture here
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

    NNValuePredictor(model, optimizer)
end

function (p::NNValuePredictor)(state)
    joint_state = reduce(vcat, state)
    p.model(joint_state)
end

function preprocess_gradients!(∇, state_value_predictor::NNValuePredictor, θ; kwargs...)
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

        # TODO: we shouldn't have to pass the annoying `action_gradient_scaling` here
        update_parameters!(state_value_predictor, ∇L; action_gradient_scaling = 1)
    end
end
