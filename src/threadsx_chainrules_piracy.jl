# Some neccesasry type piracy; adapted from:
# https://github.com/JuliaDiff/ChainRules.jl/blob/b1daa7aa6849ed66d0004de73137207e6a2d007b/src/rulesets/Base/mapreduce.jl#L74
function ChainRulesCore.rrule(
    config::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
    ::typeof(ThreadsX.map),
    f,
    xs,
)
    fx_and_pullbacks = ThreadsX.map(x -> ChainRulesCore.rrule_via_ad(config, f, x), xs)
    y = first.(fx_and_pullbacks)
    pullbacks = last.(fx_and_pullbacks)

    project = ChainRulesCore.ProjectTo(xs)
    function map_pullback(ȳ)
        f̄_and_x̄s = ThreadsX.map((f, x) -> f(x), pullbacks, ȳ)
        f̄ = if fieldcount(typeof(f)) === 0 # Then don't need to worry about derivative wrt f
            ChainRulesCore.NoTangent()
        else
            sum(first, f̄_and_x̄s)
        end
        x̄s = map(ChainRulesCore.unthunk ∘ last, f̄_and_x̄s) # project does not support receiving InplaceableThunks
        return ChainRulesCore.NoTangent(), f̄, project(x̄s)
    end
    return y, map_pullback
end
