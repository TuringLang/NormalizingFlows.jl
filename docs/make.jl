using NormalizingFlows
using Documenter

DocMeta.setdocmeta!(NormalizingFlows, :DocTestSetup, :(using NormalizingFlows); recursive=true)

makedocs(;
    modules=[NormalizingFlows],
    authors="Tor Erlend Fjelde <tor.erlend95@gmail.com> and contributors",
    repo="https://github.com/torfjelde/NormalizingFlows.jl/blob/{commit}{path}#{line}",
    sitename="NormalizingFlows.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://torfjelde.github.io/NormalizingFlows.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/torfjelde/NormalizingFlows.jl",
    devbranch="main",
)
