using CSV, DataFrames

# Read the .tsv file
dat = CSV.read("example/neural_spline_flow/data/DatasaurusDozen-Wide.tsv", DataFrame)
dat_dino = permutedims(hcat(dat.dino, dat.dino_1)[2:end, :])
dat_dino = map(x -> parse(Float32, x), dat_dino)
dat = vcat(dat_dino, dat_dino)
