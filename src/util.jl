using Functors
using Bijectors, Flux

# falttening and unflattening approach from: https://github.com/TuringLang/AdvancedVI.jl/discussions/46#discussioncomment-5543111
inner_flatten(x::Real) = [x], only
inner_flatten(x::AbstractVector{<:Real}) = x, identity
inner_flatten(x::AbstractArray{<:Real}) = vec(x), Base.Fix2(reshape, size(x))

function inner_flatten(original::Tuple)
    vecs_and_unflattens = map(inner_flatten, original)
    vecs = map(first, vecs_and_unflattens)
    unflattens = map(last, vecs_and_unflattens)

    lengths = map(length, vecs)
    end_indices = cumsum(lengths)

    N = length(original)

    function unflatten_Tuple(x)
        ntuple(N) do i
            v = original[i]
            unflatten = unflattens[i]
            l = lengths[i]
            end_idx = end_indices[i]
            start_idx = end_idx - l + 1
            return unflatten(@view(x[start_idx:end_idx]))
        end
    end

    return reduce(vcat, vecs), unflatten_Tuple
end

function inner_flatten(x::NamedTuple{names}) where {names}
    vec, re = inner_flatten(values(x))
    unflatten_NamedTuple(x) = NamedTuple{names}(re(x))
    return vec, unflatten_NamedTuple
end

"""
    flatten(x)

Return a the flattened vector of parameters and a function to reconstruct `x` from the vector.

This uses `Functors.functor` to get the parameters of `x` and `inner_flatten` to flatten them.
"""
function flatten(x)
    params, re = Functors.functor(x)
    params_vec, re_vec = inner_flatten(params)
    return params_vec, re ∘ re_vec
end

# Functors for Composed type
function Functors.functor(::Type{<:ComposedFunction}, f)
    outer, re_outer = Functors.functor(f.outer)
    inner, re_inner = Functors.functor(f.inner)
    function reconstruct_ComposedFunction(x)
        return ComposedFunction(re_outer(x.outer), re_inner(x.inner))
    end

    return (; outer, inner), reconstruct_ComposedFunction
end
