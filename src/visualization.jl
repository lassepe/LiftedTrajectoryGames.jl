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
