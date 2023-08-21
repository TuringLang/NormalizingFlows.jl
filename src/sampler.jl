# this file defines rand_device function to sample from a Distribution or a
# flow<:Bijectors.TranformedDistribution 
# this is mainly for resolving the issue of sampling from a distribution on GPU 

function rand_device(
    rng::AbstractRNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
)
    if !(rng isa CUDA.RNG)
        return nothing
    else
        return rand_cuda(rng, s)
    end
end

function rand_device(
    rng::AbstractRNG,
    s::Distributions.Sampleable{Distributions.Multivariate,Distributions.Continuous},
    n::Int,
)
    if !(rng isa CUDA.RNG)
        return Distributions.rand(rng, s, n)
    else
        return rand_cuda(rng, s, n)
    end
end

function rand_cuda(
    rng::CUDA.RNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
)
    return @inbounds Distributions.rand!(
        rng, Distributions.sampler(s), CuArray{float(eltype(s))}(undef, size(s))
    )
end

function rand_cuda(
    rng::CUDA.RNG,
    s::Distributions.Sampleable{Distributions.Multivariate,Distributions.Continuous},
    n::Int,
)
    return @inbounds Distribution.rand!(
        rng, Distributions.sampler(s), CuArray{float(eltype(s))}(undef, length(s), n)
    )
end

function Distributions._rand!(rng::CUDA.RNG, d::Distributions.MvNormal, x::CuVecOrMat)
    # Replaced usage of scalar indexing.
    Random.randn!(rng, x)
    Distributions.unwhiten!(d.Σ, x)
    x .+= d.μ
    return x
end

#########################
# for Bijectors.jl
##########################

function rand_device(rng::AbstractRNG, td::Bijectors.TransformedDistribution)
    if !(rng isa CUDA.RNG)
        return Distributions.rand(rng, td)
    else
        return cuda_rand(rng, td)
    end
end

function rand_device(rng::AbstractRNG, td::Bijectors.TransformedDistribution, n::Int)
    if !(rng isa CUDA.RNG)
        return Distributions.rand(rng, td, n)
    else
        return cuda_rand(rng, td, n)
    end
end

function cuda_rand(rng::CUDA.RNG, td::Bijectors.TransformedDistribution)
    return td.transform(cuda_rand(rng, td.dist))
end

function cuda_rand(rng::CUDA.RNG, td::Bijectors.TranformedDistribution, num_samples::Int)
    samples = cuda_rand(rng, td.dist, num_samples)
    res = reduce(
        hcat,
        map(axes(samples, 2)) do i
            return td.transform(view(samples, :, i))
        end,
    )
    return res
end
