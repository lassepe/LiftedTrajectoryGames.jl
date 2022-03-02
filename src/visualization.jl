"""
Visualize a strategy `γ` on a makie canvas using the base color `color`.

TODO: could make use of `Makie.@recipe` here.
"""
function TrajectoryGamesBase.visualize_strategy!(
    axis,
    γ::Makie.Observable{<:LiftedTrajectoryStrategy},
    color;
    weight_offset = 0.0,
)
    trajectory_colors = Makie.@lift([(color, w + weight_offset) for w in $γ.weights])
    Makie.series!(axis, γ; color = trajectory_colors)
end

function Makie.convert_arguments(::Type{<:Makie.Series}, γ::LiftedTrajectoryStrategy)
    traj_points = map(γ.trajectory_candidates) do traj
        map(s -> Makie.Point2f(s[1:2]), traj.xs)
    end
    (traj_points,)
end

# TODO: maybe this should live in the base package instead ...
function visualize_statevalue(
    env,
    state_value_predictor;
    value_levels = range(0.0, 50.0, 20),
    state_range = -3:0.1:3,
)
    color_pursuer = :red
    color_evader = :white

    fig = Makie.Figure()
    plt_ax = Makie.Axis(
        fig[1, 1];
        aspect = 1,
        xlabel = L"p_x",
        ylabel = L"p_y",
        limits = ((state_range[begin], state_range[end]), (state_range[begin], state_range[end])),
        xzoomlock = true,
        yzoomlock = true,
        xrectzoom = false,
        yrectzoom = false,
    )

    # sliders
    sl_vx1 = Makie.Slider(fig[2, 1]; range = state_range, startvalue = 0)
    Makie.Label(fig[2, 1, Makie.Left()], L"v_x^1", padding = (0, 10, 0, 0))

    sl_vx2 = Makie.Slider(fig[3, 1]; range = state_range, startvalue = 0)
    Makie.Label(fig[3, 1, Makie.Left()], L"v_x^2", padding = (0, 10, 0, 0))

    sl_vy1 = Makie.Slider(fig[1, 3]; range = state_range, startvalue = 0, horizontal = false)
    Makie.Label(fig[1, 3, Makie.Bottom()], L"v_y^1", padding = (0, 0, 0, 10))

    sl_vy2 = Makie.Slider(fig[1, 4]; range = state_range, startvalue = 0, horizontal = false)
    Makie.Label(fig[1, 4, Makie.Bottom()], L"v_y^2", padding = (0, 0, 0, 10))

    local p1

    # mouse interaction
    is_position_locked = Makie.Observable(true)
    Makie.on(Makie.events(fig).mouseposition, priority = 0) do _
        if !is_position_locked[]
            p1[] = Makie.mouseposition(plt_ax.scene)
        end
        Consume(false)
    end

    Makie.on(Makie.events(fig).mousebutton, priority = 0) do event
        if event.button == Makie.Mouse.left
            if event.action == Makie.Mouse.press
                is_position_locked[] = !is_position_locked[]
            end
        end
        Makie.Consume(false)
    end

    p1 = Makie.Observable(Makie.Point2f(-2.0, 0.0))
    v1 = Makie.@lift Makie.Vec2f($(sl_vx1.value), $(sl_vy1.value))
    v2 = Makie.@lift Makie.Vec2f($(sl_vx2.value), $(sl_vy2.value))
    z = Makie.@lift [
        (state_value_predictor([$p1..., $v1..., px2, py2, $v2...])) for px2 in state_range,
        py2 in state_range
    ]

    co = Makie.contourf!(state_range, state_range, z; levels = value_levels)
    Makie.Colorbar(fig[1, 2], co)

    # evader (P2)
    p2_positions = [Makie.Point2f(px, py) for px in -3:3, py in -3:3][:]
    p2_directions = Makie.@lift [$v2 for _ in p2_positions]
    Makie.scatter!(p2_positions; color = color_evader)
    Makie.arrows!(p2_positions, p2_directions; color = color_evader)

    # pursuer (P1)
    Makie.scatter!(p1; color = color_pursuer)
    Makie.arrows!(Makie.@lift([$p1]), Makie.@lift([$v1]); color = color_pursuer, lengthscale = 0.5)

    # environment mask
    Makie.poly!(
        Makie.Polygon(
            Makie.Point{2,Float64}[
                (state_range[begin], state_range[begin]),
                (state_range[begin], state_range[end]),
                (state_range[end], state_range[end]),
                (state_range[end], state_range[begin]),
            ],
            [GeometryBasics.coordinates(geometry(env))],
        ),
        color = :gray,
        strokewidth = 1,
    )
    fig
end
