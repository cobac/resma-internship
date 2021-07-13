using BayesianSR, GLMakie, JLD2, StatsBase

jin = load("./BayesianSR/jin/output.jld2")
simplify = load("./BayesianSR/simplify/output.jld2")
multichain = load("./BayesianSR/multichain/output.jld2")
multichain_simplify = load("./BayesianSR/multichain-simplify/output.jld2")
# evolutionary = load("./evolutionary/output.jld2")

sanitize(x::Real, max::Real = 50) = isinf(x) || isnan(x) || x > max ? convert(typeof(x), max) : x
sanitize(xs::AbstractArray, max::Real = 50) = map(x -> sanitize(x, max), xs) 

x_jin = 1:(1e5+1)
x_simplify = x_jin .* (495/336)

# sample x function x sim
# One sample chain
lines(x_jin, sanitize(jin["rmse_train"][:, 1, 1]))

# sample x chain x function x sim
# One sample multichain
y = multichain["rmse_train"][:, :, 1, 1]
lines(x_jin, y[:, 1], linewidth = 0.5)
for chain in 2:4
    lines!(x_jin, y[:, chain], linewidth = 0.5)
end 

# Averaged multichain chains
multichain_meany = mean(collect(eachslice(sanitize(multichain["rmse_train"]), dims=2)))
multichain_simplify_meany = mean(collect(eachslice(sanitize(multichain_simplify["rmse_train"]), dims=2)))

lines(x_jin, multichain_meany[:, 1, 1], linewidth = 0.5)

lines(x_jin, multichain_meany[:, 1, 2], linewidth = 0.5)

lines(x_jin, multichain_meany[:, 6, 1], linewidth = 0.5)

# Original average over simulations
lines(x_jin, mean.(eachrow(sanitize(jin["rmse_train"][:, 1, :]))), linewidth = 0.5)

lines(x_jin, mean.(eachrow(sanitize(jin["rmse_train"][:, 6, :]))), linewidth = 0.5)

# Multichain average over simulations
lines(x_jin, mean.(eachrow(multichain_meany[:, 1, :])), linewidth = 0.5)

lines(x_jin, mean.(eachrow(multichain_meany[:, 6, :])), linewidth = 0.5)

# Simplify
lines(x_jin, mean.(eachrow(sanitize(jin["rmse_train"][:, 1, :]))), linewidth = 0.5)
lines!(x_simplify, mean.(eachrow(sanitize(simplify["rmse_train"][:, 1, :]))), linewidth = 0.5)

# Test dataset
lines(x_jin, mean.(eachrow(sanitize(jin["rmse_train"][:, 2, :]))), linewidth = 0.5)
lines!(x_jin, mean.(eachrow(sanitize(jin["rmse_test₁"][:, 2, :]))), linewidth = 0.5)

positions = CartesianIndices(zeros(3,2))

# Original vs simplify
fig = Figure()
for eq in 1:6
    ax = fig[Tuple(positions[eq])...] = Axis(fig, title = string("Equation no. ", eq))
    lines!(ax, x_jin, mean.(eachrow(sanitize(jin["rmse_train"][:, eq, :]))), linewidth = 0.5, label = "Original")
    lines!(ax, x_simplify, mean.(eachrow(sanitize(simplify["rmse_train"][:, eq, :]))), linewidth = 0.5, label = "Simplify step")
end 
axislegend()

# Original vs simplify cummin
fig = Figure()
for eq in 1:6
    ax = fig[Tuple(positions[eq])...] = Axis(fig, title = string("Equation no. ", eq))
    lines!(ax, x_jin, accumulate(min, mean.(eachrow(sanitize(jin["rmse_train"][:, eq, :])))), linewidth = 1.5, label = "Original")
    lines!(ax, x_simplify, accumulate(min, mean.(eachrow(sanitize(simplify["rmse_train"][:, eq, :])))), linewidth = 1.5, label = "Simplify step")
end 
axislegend()

# Original vs simplify vs multichain
fig = Figure()
for eq in 1:6
    ax = fig[Tuple(positions[eq])...] = Axis(fig, title = string("Equation no. ", eq))
    lines!(ax, x_jin, mean.(eachrow(sanitize(jin["rmse_train"][:, eq, :]))), linewidth = 0.5, label = "Original")
    lines!(ax, x_simplify, mean.(eachrow(sanitize(simplify["rmse_train"][:, eq, :]))), linewidth = 0.5, label = "Simplify step")
    lines!(ax, x_jin, mean.(eachrow(multichain_meany[:, 1, :])), linewidth = 0.5, label = "Multichain")
end 
axislegend()


# Original vs simplify vs multichain vs multichain+simplify
fig = Figure()
for eq in 1:6
    ax = fig[Tuple(positions[eq])...] = Axis(fig, title = string("Equation no. ", eq))
    lines!(ax, x_jin, mean.(eachrow(sanitize(jin["rmse_train"][:, eq, :]))), linewidth = 0.5, label = "Original")
    lines!(ax, x_simplify, mean.(eachrow(sanitize(simplify["rmse_train"][:, eq, :]))), linewidth = 0.5, label = "Simplify step")
    lines!(ax, x_jin, mean.(eachrow(multichain_meany[:, 1, :])), linewidth = 0.5, label = "Multichain")
    lines!(ax, x_simplify, mean.(eachrow(multichain_simplify_meany[:, 1, :])), linewidth = 0.5, label = "Multichain simplify")
end 
axislegend()

# Focus on multichain
fig = Figure()
for eq in 1:6
    ax = fig[Tuple(positions[eq])...] = Axis(fig, title = string("Equation no. ", eq))
    lines!(ax, x_jin, mean.(eachrow(multichain_meany[:, 1, :])), linewidth = 0.5, label = "Multichain")
    lines!(ax, x_simplify, mean.(eachrow(multichain_simplify_meany[:, 1, :])), linewidth = 0.5, label = "Multichain simplify")
end 
axislegend()

# Multichain with test
multichain_meany_test = mean(collect(eachslice(sanitize(multichain["rmse_test₁"], 500), dims=2)))
fig = Figure()
for eq in 1:6
    ax = fig[Tuple(positions[eq])...] = Axis(fig, title = string("Equation no. ", eq))
    lines!(ax, x_jin, mean.(eachrow(multichain_meany[:, 1, :])), linewidth = 0.5, label = "Train")
    lines!(ax, x_jin, mean.(eachrow(multichain_meany_test[:, 1, :])), linewidth = 0.5, label = "Test")
end 
axislegend()
