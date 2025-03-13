function rand_device(
    rng::Random.AbstractRNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
)
    return Random.rand(rng, s)
end

function rand_device(
    rng::Random.AbstractRNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
    n::Int,
)
    return Random.rand(rng, s, n)
end

function rand_device(rng::Random.AbstractRNG, td::Bijectors.TransformedDistribution)
    return Random.rand(rng, td)
end

function rand_device(rng::Random.AbstractRNG, td::Bijectors.TransformedDistribution, n::Int)
    return Random.rand(rng, td, n)
end
