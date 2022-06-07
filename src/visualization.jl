"""
Visualize a strategy `γ` on a makie canvas using the base color `color`.
"""
function TrajectoryGamesBase.visualize!(
    canvas,
    γ::Makie.Observable{<:LiftedTrajectoryStrategy};
    color = :black,
    weight_offset = 0.0,
)
    trajectory_colors = Makie.@lift([(color, w + weight_offset) for w in $γ.weights])
    Makie.series!(canvas, γ; color = trajectory_colors)
end

function Makie.convert_arguments(::Type{<:Makie.Series}, γ::LiftedTrajectoryStrategy)
    traj_points = map(γ.trajectories) do traj
        map(s -> Makie.Point2f(s[1:2]), traj.xs)
    end
    (traj_points,)
end
