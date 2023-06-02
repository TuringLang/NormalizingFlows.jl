using NormalizingFlows
using Documenter

DocMeta.setdocmeta!(
    NormalizingFlows, :DocTestSetup, :(using NormalizingFlows); recursive=true
)

makedocs(;
    modules=[NormalizingFlows],
    repo="https://github.com/TuringLang/NormalizingFlows.jl/blob/{commit}{path}#{line}",
    sitename="NormalizingFlows.jl",
    format=Documenter.HTML(),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/TuringLang/NormalizingFlows.jl", devbranch="main")
