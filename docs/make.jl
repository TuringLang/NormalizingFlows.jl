using NormalizingFlows
using Documenter

using Random
using Distributions

DocMeta.setdocmeta!(
    NormalizingFlows, :DocTestSetup, :(using NormalizingFlows); recursive=true
)

makedocs(;
    modules=[NormalizingFlows],
    sitename="NormalizingFlows.jl",
    repo="https://github.com/TuringLang/NormalizingFlows.jl/blob/{commit}{path}#{line}",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true"),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
        "Example" => "example.md",
        "Customize your own flow layer" => "customized_layer.md",
    ],
    checkdocs=:exports,
)
