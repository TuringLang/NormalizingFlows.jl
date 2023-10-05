using Functors
using Bijectors

struct InvertibleMLP{T1,T2} <: Bijectors.Bijector
    shift::T1
    scale::T2
    invertible_act::Bijectors.Bijector
end

@functor InvertibleMLP