# Import RDatasets.
using RDatasets
# Functionality for splitting and normalizing the data
using MLDataUtils: rescale!


# Import the "Default" dataset.
data = RDatasets.dataset("ISLR", "Default");
# Convert "Default" and "Student" to numeric values.
data[!,:DefaultNum] = [r.Default == "Yes" ? 1.0 : 0.0 for r in eachrow(data)]
data[!,:StudentNum] = [r.Student == "Yes" ? 1.0 : 0.0 for r in eachrow(data)]
# Delete the old columns which say "Yes" and "No".
select!(data, Not([:Default, :Student]))
# using frist 1000 datapoints
data = data[1:1000,:]

# data processing
features = [:StudentNum, :Balance, :Income]
numerics = [:Balance, :Income]
target = :DefaultNum
for feature in numerics
  μ, σ = rescale!(data[!, feature], obsdim=1)
  rescale!(data[!, feature], μ, σ, obsdim=1)
end

# Turing requires data in matrix form, not dataframe
X = Matrix(data[:, features])
Y = data[:, target]


# saving feature and label
save("example/logistic_reg/data/dataset.jld", "X", X, "Y", Y)

dataset = load("example/logistic_reg/data/dataset.jld")