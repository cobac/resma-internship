using BayesianSR, CairoMakie, JLD2, StatsBase

jin = load("./BayesianSR/jin/output.jld2")
simplify = load("./BayesianSR/simplify/output.jld2")
multichain = load("./BayesianSR/multichain/output.jld2")
multichain_simplify = load("./BayesianSR/multichain-simplify/output.jld2")
evolutionary = load("./evolutionary/output.jld2")

# Cap maximum RMSE to a high value to plot infinities and extremely high values
sanitize(x::Real, max::Real) = isinf(x) || isnan(x) || x > max ? convert(typeof(x), max) : x
sanitize(xs::AbstractArray, max::Real = 500) = map(x -> sanitize(x, max), xs) 

x_jin = 1:(1e5+1)
# Iterations adjusted to our custom time units
x_simplify = x_jin .* (495/336)

# Average RMSE values across all chains for multi-chain algorithms
multichain_meany = mean(collect(eachslice(sanitize(multichain["rmse_train"]), dims=2)))
multichain_simplify_meany = mean(collect(eachslice(sanitize(multichain_simplify["rmse_train"]), dims=2)))

my_theme = Theme(Axis = (xlabel = "Iterations", ylabel = "RMSE",
                         topspinevisible = false, rightspinevisible = false, xgridvisible = false),
                 ScatterLines = (markersize = 3,))
colors = Makie.wong_colors()
# One example chain

double_text_size = 32
ex_chain = with_theme(my_theme) do
    ex_chain = lines(x_jin, sanitize(jin["rmse_train"][:, 1, 1]), color = colors[1],
                     axis = (xticks = LinearTicks(3),
                             xlabelsize = double_text_size, ylabelsize = double_text_size,
                             xticklabelsize = double_text_size, yticklabelsize = double_text_size))
    ex_chain
end 
save("./figures/ex_chain.svg", ex_chain)

# One sample multichain
y = multichain["rmse_train"][:, :, 1, 1]
x_until = 15000
ex_multichain = with_theme(my_theme) do
    ex_multichain = lines(x_jin[1:x_until], y[1:x_until, 1],
                          axis = (xticks = LinearTicks(3),
                                  xlabelsize = double_text_size, ylabelsize = double_text_size,
                                  xticklabelsize = double_text_size, yticklabelsize = double_text_size))
    for chain in 2:4
        lines!(x_jin[1:x_until], y[1:x_until, chain])
    end 
    ex_multichain
end 
save("./figures/ex_multichain.svg", ex_multichain)

grid_positions = CartesianIndices(zeros(3,2))

# Legend elements

leg_lw = 4
leg_original = LineElement(linewidth = leg_lw, color = colors[1])
leg_simplify = LineElement(linewidth = leg_lw, color = colors[2])
leg_multichain = LineElement(linewidth = leg_lw, color = colors[3])
leg_multichain_simplify = LineElement(linewidth = leg_lw, color = colors[4])
leg_test₁ = LineElement(linewidth = leg_lw, color = colors[5])
leg_test₂ = LineElement(linewidth = leg_lw, color = colors[6])
leg_test₃ = LineElement(linewidth = leg_lw, color = colors[7])
                          

# Original vs simplify
original_simplify = with_theme(my_theme) do
    original_simplify = Figure(resolution = (1000, 1200))
    lw = 1
    axs = Vector{Axis}(undef, 6)
    for eq in 1:6
        axs[eq] = original_simplify[Tuple(grid_positions[eq])...] = Axis(original_simplify, title = string("Equation no. ", eq),
                                                                         yminorticksvisible = false, yticks = LinearTicks(2), xticks = LinearTicks(3))
        lines!(axs[eq], x_jin, mean.(eachrow(sanitize(jin["rmse_train"][:, eq, :]))),
               color = colors[1], label = "Original", linewidth = lw)
               lines!(axs[eq], x_simplify, mean.(eachrow(sanitize(simplify["rmse_train"][:, eq, :]))),
                      color = colors[2], label = "Simplify step", linewidth = lw)

    end 
    linkxaxes!(axs[1], axs[2], axs[3])
    linkxaxes!(axs[4], axs[5], axs[6])
    for i in [1,2,4,5]
        hidexdecorations!(axs[i], grid  = false)
    end 
    original_simplify[4, 1:2] = Legend(original_simplify,
                                       [leg_original, leg_simplify],
                                       ["Original", "Simplify step"],
                                       tellheight = true, framevisible = false, orientation = :horizontal)
    original_simplify
end 
save("./figures/original_simplify.svg", original_simplify)


# Original vs multi-chain
original_multichain = with_theme(my_theme) do
    original_multichain = Figure(resolution = (1000, 1200))
    lw = 1
    axs = Vector{Axis}(undef, 6)
    for eq in 1:6
        axs[eq] = original_multichain[Tuple(grid_positions[eq])...] = Axis(original_multichain, title = string("Equation no. ", eq),
                                                                           yminorticksvisible = false, yticks = LinearTicks(2), xticks = LinearTicks(3))
        lines!(axs[eq], x_jin, mean.(eachrow(sanitize(jin["rmse_train"][:, eq, :]))),
               color = colors[1], linewidth = lw, label = "Original")
        lines!(axs[eq], x_jin, mean.(eachrow(multichain_meany[:, eq, :])),
               color = colors[3], linewidth = lw, label = "Multichain")
    end 
    linkxaxes!(axs[1], axs[2], axs[3])
    linkxaxes!(axs[4], axs[5], axs[6])
    for i in [1,2,4,5]
        hidexdecorations!(axs[i], grid  = false)
    end 
    original_multichain[4, 1:2] = Legend(original_multichain,
                                         [leg_original, leg_multichain],
                                         ["Original", "Multichain sampling"],
                                         tellheight = true, framevisible = false, orientation = :horizontal)

    original_multichain
end 
save("./figures/original_multichain.svg", original_multichain)


# Multichain vs multichain+simplify
multichain_simplify = with_theme(my_theme) do
    multichain_simplify = Figure(resolution = (1000, 1200))
    lw = 1
    axs = Vector{Axis}(undef, 6)
    for eq in 1:6
        axs[eq] = multichain_simplify[Tuple(grid_positions[eq])...] = Axis(multichain_simplify, title = string("Equation no. ", eq),
                                                                           yminorticksvisible = false, yticks = LinearTicks(2), xticks = LinearTicks(3))
        lines!(axs[eq], x_jin, mean.(eachrow(multichain_meany[:, eq, :])),
               color = colors[3], linewidth = lw, label = "Multichain")
        lines!(axs[eq], x_simplify, mean.(eachrow(multichain_simplify_meany[:, eq, :])),
               color = colors[4], linewidth = lw, label = "Multichain+simplify")
        eq == 1 && ylims!(0, 10)
        eq == 2 && ylims!(0, 15)
        eq == 3 && ylims!(0, 20)
        eq == 4 && ylims!(0, 5)
        eq == 5 && ylims!(0, 25)
        eq == 6 && ylims!(0, 35)
    end 
    linkxaxes!(axs[1], axs[2], axs[3])
    linkxaxes!(axs[4], axs[5], axs[6])
    for i in [1,2,4,5]
        hidexdecorations!(axs[i], grid  = false)
    end 
    multichain_simplify[4, 1:2] = Legend(multichain_simplify,
                                         [leg_multichain, leg_multichain_simplify],
                                         ["Multichain sampling", "Multichain sampling + simplification step"],
                                         tellheight = true, framevisible = false, orientation = :horizontal)
    multichain_simplify
end 
save("./figures/multichain_simplify.svg", multichain_simplify)

# Multichain with test
multichain_meany_test₁ = mean(collect(eachslice(sanitize(multichain["rmse_test₁"], 2000), dims=2)))
multichain_meany_test₂ = mean(collect(eachslice(sanitize(multichain["rmse_test₂"], 2000), dims=2)))
multichain_meany_test₃ = mean(collect(eachslice(sanitize(multichain["rmse_test₃"], 2000), dims=2)))

multichain_tests = with_theme(my_theme) do
    multichain_tests = Figure(resolution = (1000, 1200))
    lw = 1 
    axs = Vector{Axis}(undef, 6)
    for eq in 1:6
        axs[eq] = multichain_tests[Tuple(grid_positions[eq])...] = Axis(multichain_tests, title = string("Equation no. ", eq))
        lines!(axs[eq], x_jin, mean.(eachrow(multichain_meany[:, eq, :])), linewidth = lw, label = "Train set U(-3, 3)", color = colors[3])
        lines!(axs[eq], x_jin, mean.(eachrow(multichain_meany_test₁[:, eq, :])), linewidth = lw, label = "Test set U(-3, 3)", color = colors[5])
        lines!(axs[eq], x_jin, mean.(eachrow(multichain_meany_test₂[:, eq, :])), linewidth = lw, label = "Test set U(-6, 6)", color = colors[6])
        lines!(axs[eq], x_jin, mean.(eachrow(multichain_meany_test₃[:, eq, :])), linewidth = lw, label = "Test set U(3, 6)", color = colors[7])
    end 
    linkxaxes!(axs[1], axs[2], axs[3])
    linkxaxes!(axs[4], axs[5], axs[6])
    for i in [1,2,4,5]
        hidexdecorations!(axs[i], grid  = false)
    end 

    multichain_tests[4, 1:2] = Legend(multichain_tests,
                                      [leg_multichain, leg_test₁, leg_test₂, leg_test₃],
                                      ["Train set U(-3, 3)", "Test set U(-3, 3)",  "Test set U(-6, 6)", "Test set U(3, 6)" ],
                                      tellheight = true, framevisible = false, orientation = :horizontal)
    multichain_tests
end 
save("./figures/multichain_tests.svg", multichain_tests)

original_tests = with_theme(my_theme) do
    original_tests = Figure(resolution = (1000, 1200))
    lw = 1 
    axs = Vector{Axis}(undef, 6)
    for eq in 1:6
        axs[eq] = original_tests[Tuple(grid_positions[eq])...] = Axis(original_tests, title = string("Equation no. ", eq))
        lines!(axs[eq], x_jin, mean.(eachrow(sanitize(jin["rmse_train"][:, eq, :], 2000))), linewidth = lw, label = "Train set U(-3, 3)", color = colors[1])
        lines!(axs[eq], x_jin, mean.(eachrow(sanitize(jin["rmse_test₁"][:, eq, :], 2000))), linewidth = lw, label = "Test set U(-3, 3)", color = colors[5])
        lines!(axs[eq], x_jin, mean.(eachrow(sanitize(jin["rmse_test₂"][:, eq, :], 2000))), linewidth = lw, label = "Test set U(-6, 6)", color = colors[6])
        lines!(axs[eq], x_jin, mean.(eachrow(sanitize(jin["rmse_test₃"][:, eq, :], 2000))), linewidth = lw, label = "Test set U(3, 6)", color = colors[7])
    end 
    linkxaxes!(axs[1], axs[2], axs[3])
    linkxaxes!(axs[4], axs[5], axs[6])
    for i in [1,2,4,5]
        hidexdecorations!(axs[i], grid  = false)
    end 

    original_tests[4, 1:2] = Legend(original_tests,
                                    [leg_original, leg_test₁, leg_test₂, leg_test₃],
                                    ["Train set U(-3, 3)", "Test set U(-3, 3)",  "Test set U(-6, 6)", "Test set U(3, 6)" ],
                                      tellheight = true, framevisible = false, orientation = :horizontal)
    original_tests
end 
save("./figures/original_tests.svg", original_tests)


hof = with_theme(my_theme) do
    hof = Figure(resolutioon = (1000, 1200))
    lw = 2
    axs = Vector{Axis}(undef, 6)
    for eq in 1:6
        axs[eq] = hof[Tuple(grid_positions[eq])...] = Axis(hof, title = string("Equation no. ", eq),
                                                           xlabel = "Complexity", xticksvisible = false, xticklabelsvisible = false)
        y_train = filter(!iszero, evolutionary["rmse_train"][:, eq])
        y_test₁ = filter(!iszero, evolutionary["rmse_test₁"][:, eq])
        y_test₂ = filter(!iszero, evolutionary["rmse_test₂"][:, eq])
        y_test₃ = filter(!iszero, evolutionary["rmse_test₃"][:, eq])
        scatterlines!(axs[eq], 1:length(y_train), y_train, linewidth = lw, color = colors[1], markercolor = colors[1])
        scatterlines!(axs[eq], 1:length(y_test₁), y_test₁, linewidth = lw, color = colors[5], markercolor = colors[5])
        scatterlines!(axs[eq], 1:length(y_test₂), y_test₂, linewidth = lw, color = colors[6], markercolor = colors[6])
        scatterlines!(axs[eq], 1:length(y_test₃), y_test₃, linewidth = lw, color = colors[7], markercolor = colors[7])
        eq == 2 && ylims!(0, 1500)
    end 

    #linkxaxes!(axs[1], axs[2], axs[3])
    #linkxaxes!(axs[4], axs[5], axs[6])
    for i in [1,2,4,5]
        hidexdecorations!(axs[i])
    end 
    hof[4, 1:2] = Legend(hof,
                         [leg_original, leg_test₁, leg_test₂, leg_test₃],
                         ["Train set U(-3, 3)", "Test set U(-3, 3)",  "Test set U(-6, 6)", "Test set U(3, 6)" ],
                         tellheight = true, framevisible = false, orientation = :horizontal)
    hof
end 
save("./figures/evolutionary.svg", hof)
