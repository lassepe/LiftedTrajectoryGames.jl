using LiftedTrajectoryGames: LiftedTrajectoryGames
using Test: @test, @testset
using Zygote: Zygote
using ThreadsX: ThreadsX
using Base.Threads: nthreads

@testset "LiftedTrajectoryGames" begin
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

    @testset "ThreadsX ChainRules exentions" begin
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
