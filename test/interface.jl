@testset "testing mean-field Gaussian VI" begin
    chunksize = 4
    @testset "$adtype" for adtype in [
        ADTypes.AutoZygote(),
        ADTypes.AutoForwardDiff(; chunksize=chunksize),
        ADTypes.AutoForwardDiff(),
        ADTypes.AutoReverseDiff(),
        ADTypes.AutoEnzyme(;
            mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation=Enzyme.Const,
        ),
        ADTypes.AutoMooncake(; config = Mooncake.Config()), 
    ]
        @testset "$T" for T in [Float32, Float64]
            μ = 10 * ones(T, 2)
            Σ = Diagonal(4 * ones(T, 2))

            target = MvNormal(μ, Σ)
            logp(z) = logpdf(target, z)

            @leaf MvNormal
            q₀ = MvNormal(zeros(T, 2), ones(T, 2))
            flow = Bijectors.transformed(
                q₀, Bijectors.Shift(zeros(T, 2)) ∘ Bijectors.Scale(ones(T, 2))
            )

            sample_per_iter = 10
            cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,ad=adtype)
            checkconv(iter, stat, re, θ, st) = stat.gradient_norm < one(T)/1000
            flow_trained, stats, _ = train_flow(
                elbo,
                flow,
                logp,
                sample_per_iter;
                max_iters=5_000,
                optimiser=Optimisers.Adam(one(T)/100),
                ADbackend=adtype,
                show_progress=false,
                callback=cb,
                hasconverged=checkconv,
            )
            θ, re = Optimisers.destructure(flow_trained)

            el_untrained = elbo(flow, logp, 1000)
            el_trained = elbo(flow_trained, logp, 1000)

            @test all(abs.(θ[1:2] .- μ) .< 0.2)
            @test all(abs.(θ[3:4] .- 2) .< 0.2)
            @test el_trained > el_untrained
            @test el_trained > -1
        end
    end
end

# function create_planar_flow(n_layers::Int, q₀, T)
#     d = length(q₀)
#     if T == Float32
#         Ls = reduce(∘, [f32(PlanarLayer(d)) for _ in 1:n_layers])
#     else
#         Ls = reduce(∘, [PlanarLayer(d) for _ in 1:n_layers])
#     end
#     return Bijectors.transformed(q₀, Ls)
# end

# @testset "testing planar flow" begin
#     chunksize = 4
#     @testset "$adtype" for adtype in [
#         ADTypes.AutoZygote(),
#         ADTypes.AutoForwardDiff(; chunksize=chunksize),
#         ADTypes.AutoForwardDiff(),
#         ADTypes.AutoReverseDiff(),
#         ADTypes.AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse)),
#         ADTypes.AutoMooncake(; config = Mooncake.Config()), # somehow Mooncake does not work with Float64
#     ]
#         @testset "$T" for T in [Float32, Float64]
#             μ = 10 * ones(T, 2)
#             Σ = Diagonal(4 * ones(T, 2))

#             target = MvNormal(μ, Σ)
#             logp(z) = logpdf(target, z)

#             @leaf MvNormal
#             q₀ = MvNormal(zeros(T, 2), ones(T, 2))
#             nlayers = 10
#             flow = create_planar_flow(nlayers, q₀, T)

#             sample_per_iter = 10
#             cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,ad=adtype)
#             checkconv(iter, stat, re, θ, st) = stat.gradient_norm < one(T)/1000
#             flow_trained, stats, _, _ = train_flow(
#                 elbo,
#                 flow,
#                 logp,
#                 sample_per_iter;
#                 max_iters=10_000,
#                 optimiser=Optimisers.Adam(one(T)/100),
#                 ADbackend=adtype,
#                 show_progress=false,
#                 callback=cb,
#                 hasconverged=checkconv,
#             )
#             θ, re = Optimisers.destructure(flow_trained)

#             el_untrained = elbo(flow, logp, 1000)
#             el_trained = elbo(flow_trained, logp, 1000)

#             @test el_trained > el_untrained
#             @test el_trained > -1
#         end
#     end
# end
