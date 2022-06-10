# LiftedTrajectoryGames

A reference implementation of the lifted game solver presented in our RSS 2022 paper ["Learning Mixed Strategies in Trajectory Games"](https://arxiv.org/pdf/2205.00291.pdf).

## Paper

[![](https://lasse-peters.net/static/img/peters2022rss-teaser.png)](https://arxiv.org/pdf/2205.00291.pdf)

```bibtex
@inproceedings{peters2021rss,
    title     = {Learning Mixed Strategies in Trajectory Games},
    author    = {Peters, Lasse and Fridovich-Keil, David and Ferranti, Laura and Stachniss, Cyrill and Alonso-Mora, Javier and Laine, Forrest},
    booktitle = {Proc.~of Robotics: Science and Systems (RSS)},
    year      = {2022},
    url       = {https://arxiv.org/abs/2106.03611}
}
```

## Installation

> :warning: LiftedTrajectoryGames is not yet registered in the General registry. For now, you have to add it manually via the package URL: `pkg> add https://github.com/lassepe/LiftedTrajectoryGames.jl`

To install LiftedTrajectoryGames, simply add it via Julia's package manager from the REPL:

```julia
# hit `]` to enter "pkg"-mode of the REPL
pkg> add LiftedTrajectoryGames
```

## Usage

LiftedTrajectoryGames uses [TrajectoryGamesBase](https://github.com/lassepe/TrajectoryGamesBase.jl) as an abstraction of the problem and solver interface for trajectory games. Please to refer to that package for documentation on the problem setup. Note that the lifted game solver **requires differentiability of the game's costs and dynamics**.

For a game that meets those assumptions, you can construct a `solver::LiftedTrajectoryGameSolver` using the helper constructor that recovers the relevant solver parameters (network input/output dimensions etc.) from a given `game::TrajectoryGame`. Please refer to the docstring of the `LiftedTrajectoryGameSolver` for a more complete description of the solver options.

```julia
using LiftedTrajectoryGames
using TrajectoryGamesBase
using Random: MersenneTwister

# place holder; replace with your actual game constructor
game = construct_your_game_of_choice()

planning_horizon = 20
# the number of "pure" trajectories to mix over for each player:
n_actions = [2 for _ in 1:num_players(game)]
solver = LiftedTrajectoryGames(solver, planning_horizon; n_actions)
```

> :warning: Note that the solver construction may take a while as it compiles all the relevant functions and derivatives for acceleration of downstream solver invocations.

Once you have set up the solver, you can invoke it for a given `initial_state`.

```julia
strategy = solve_trajectory_game!(solver, game, initial_state)
```

The resulting *mixed* joint `strategy` can then be invoked on the state to compute `controls` for all players.

```julia
# A strategy may be time-varying. Therefore, we also have to hand in the time.
time = 1
controls = strategy(initial_state, time)
```
> :warning: TODO:
> - load example problem from somewhere to have a copy-pastable example here.
> - demonstrate learning from scratch in a receding-horizon setting
> - explain learning settings
