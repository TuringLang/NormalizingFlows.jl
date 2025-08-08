using NormalizingFlows
using Documenter

DocMeta.setdocmeta!(
    NormalizingFlows, :DocTestSetup, :(using NormalizingFlows); recursive=true
)

makedocs(;
    modules=[NormalizingFlows],
    sitename="NormalizingFlows.jl",
    format=Documenter.HTML(;
        repolink="https://github.com/TuringLang/NormalizingFlows.jl/blob/{commit}{path}#{line}",
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
        "Example" => "example.md",
        "Customize your own flow layer" => "customized_layer.md",
    ],
    checkdocs=:exports,
)