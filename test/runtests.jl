using Pkg: Pkg
Pkg.develop(path="../../TrajectoryGamesExamples")


using LiftedTrajectoryGames:
    LiftedTrajectoryGames,
    LiftedTrajectoryGameSolver,
    MultiThreadedExecutionPolicy,
    NeuralStateValuePredictor
using Test: @test, @testset
using Zygote: Zygote
using ThreadsX: ThreadsX
using Base.Threads: nthreads

using TrajectoryGamesBase: RecedingHorizonStrategy, num_players, rollout, solve_trajectory_game!
using TrajectoryGamesExamples: animate_sim_steps, two_player_meta_tag
using BlockArrays: mortar
using Random: MersenneTwister

@testset "LiftedTrajectoryGames" begin
    @testset "Solver integration tests" begin
        for (description, game) in [("meta tag", two_player_meta_tag())]
            @testset "$description" begin
                if num_players(game) == 2
                    initial_state = mortar([[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
                elseif num_players(game) == 3
                    initial_state =
                        mortar([[0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
                else
                    error("No config for games of this size.")
                end

                planning_horizon = 20
                rng = MersenneTwister(1)
                turn_length = 10

                @testset "Learning" begin
                    solver = LiftedTrajectoryGameSolver(
                        game,
                        planning_horizon;
                        gradient_clipping_threshold = nothing,
                    )
                    solve_trajectory_game!(solver, game, initial_state)
                end

                @testset "Gradient Clipping" begin
                    solver = LiftedTrajectoryGameSolver(
                        game,
                        planning_horizon;
                        gradient_clipping_threshold = 0.1,
                    )
                    solve_trajectory_game!(solver, game, initial_state)
                end

                @testset "State value prediction" begin
                    state_value_predictor =
                        NeuralStateValuePredictor(; game, rng, turn_length, learning_rate = 0.02)
                    solver =
                        LiftedTrajectoryGameSolver(game, planning_horizon; state_value_predictor)
                    solve_trajectory_game!(solver, game, initial_state)
                end

                @testset "Multi-Threaded Execution" begin
                    solver = LiftedTrajectoryGameSolver(
                        game,
                        planning_horizon;
                        execution_policy = MultiThreadedExecutionPolicy(),
                    )
                    solve_trajectory_game!(solver, game, initial_state)
                end

                @testset "Receding horizon" begin
                    solver = LiftedTrajectoryGameSolver(
                        game,
                        planning_horizon;
                        execution_policy = MultiThreadedExecutionPolicy(),
                    )
                    receding_horizon_strategy =
                        RecedingHorizonStrategy(; solver, game, turn_length = 10)

                    local sim_steps

                    @testset "rollout" begin
                        sim_steps = rollout(
                            game.dynamics,
                            receding_horizon_strategy,
                            initial_state,
                            100;
                            get_info = (γ, x, t) -> γ.receding_horizon_strategy,
                        )
                    end

                    @testset "animation" begin
                        animate_sim_steps(game, sim_steps; live = false, framerate = 30)
                    end
                end
            end
        end
    end

    @testset "ThreadsX ChainRules exentions" begin
        if !(nthreads() > 1)
            error("These tests are not meaningful without multi-threading enabled.")
        end

        function f(x)
            sleep(1)
            x^2
        end

        function plain_base_map(itr)
            mapped_itr = map(f, itr)
            first(mapped_itr)
        end

        function plain_threadsx_map(itr)
            mapped_itr = ThreadsX.map(f, itr)
            first(mapped_itr)
        end

        function closure_base_map(itr)
            map(itr) do i
                f(itr[i])
            end
        end

        function closure_threadsx_map(itr)
            ThreadsX.map(itr) do i
                f(itr[i])
            end
        end

        x = [1, 2, 3]
        @testset "plain map" begin
            # called twice to trigger compilation before timing
            Zygote.jacobian(plain_base_map, x)
            res_base = @timed Zygote.jacobian(plain_base_map, x)

            # called twice to trigger compilation before timing
            Zygote.jacobian(plain_threadsx_map, x)
            res_threadsx = @timed Zygote.jacobian(plain_threadsx_map, x)

            # Note: To pass, these tests must be spawned with multiple threads
            @test res_threadsx.time < res_base.time / length(x) + 0.1
            @test res_threadsx.value == res_base.value
        end

        @testset "closure map" begin
            # called twice to trigger compilation before timing
            Zygote.jacobian(closure_base_map, x)
            res_base = @timed Zygote.jacobian(closure_base_map, x)

            # called twice to trigger compilation before timing
            Zygote.jacobian(closure_threadsx_map, x)
            res_threadsx = @timed Zygote.jacobian(closure_threadsx_map, x)

            # Note: To pass, these tests must be spawned with multiple threads
            @test res_threadsx.time < res_base.time / length(x) + 0.1
            @test res_threadsx.value == res_base.value
        end
    end
end
