using BayesianSR,
    Random,
    LinearAlgebra,
    StatsBase,
    Distributions,
    ExprRules,
    JLD

chains = []
t = []

for file in readdir("./out")
    l = load(string("./out/", file))
    push!(chains, l["chains"])
    push!(t, l["t"])
end 

x = chains[1][1].x

Random.seed!(1)
x_test₁ = rand(Uniform(-3 , 3), (30, 2))
x_test₂ = rand(Uniform(-6 , 6), (30, 2))
x_test₃ = rand(Uniform(3 , 6),  (30, 2))

accepted₁ = 0
accepted₂ = 0
accepted₃ = 0
accepted₄ = 0
accepted₅ = 0
accepted₆ = 0
for i in eachindex(chains)
    accepted₁ +=  chains[i][1].stats[:accepted]
    accepted₂ +=  chains[i][2].stats[:accepted]
    accepted₃ +=  chains[i][3].stats[:accepted]
    accepted₄ +=  chains[i][4].stats[:accepted]
    accepted₅ +=  chains[i][5].stats[:accepted]
    accepted₆ +=  chains[i][6].stats[:accepted]
end 

rate₁ = accepted₁ / length(chains[1][1]) / length(chains)
rate₂ = accepted₂ / length(chains[1][2]) / length(chains)
rate₃ = accepted₃ / length(chains[1][3]) / length(chains)
rate₄ = accepted₄ / length(chains[1][4]) / length(chains)
rate₅ = accepted₅ / length(chains[1][5]) / length(chains)
rate₆ = accepted₆ / length(chains[1][6]) / length(chains)

eq₁ = get_function(chains[1][1], latex = false)
eq₂ = get_function(chains[2][2], latex = false)
eq₃ = get_function(chains[3][3], latex = false)
eq₄ = get_function(chains[4][4], latex = false)
eq₅ = get_function(chains[5][5], latex = false)
eq₆ = get_function(chains[6][6], latex = false)

# 3D Array with dimensions samples/chain X no_chains X no_simulations
rmse_train = Array{Float64}(undef, length(chains[1][1]),
                            length(chains[1]),
                            length(chains))
rmse_test₁ = Array{Float64}(undef, length(chains[1][1]),
                            length(chains[1]),
                            length(chains))
rmse_test₂ = Array{Float64}(undef, length(chains[1][1]),
                            length(chains[1]),
                            length(chains))
rmse_test₃ = Array{Float64}(undef, length(chains[1][1]),
                            length(chains[1]),
                            length(chains))

for sample=1:length(chains[1][1]),
    chain=1:length(chains[1]),
    sim=1:length(chains)
    y = chains[sim][chain].y
    ŷ_train = evalmodel(chains[sim][chain].samples[sample],
                        x,
                        chains[sim][chain].grammar) 
    ŷ_test₁ = evalmodel(chains[sim][chain].samples[sample],
                        x_test₁,
                        chains[sim][chain].grammar) 
    ŷ_test₂ = evalmodel(chains[sim][chain].samples[sample],
                        x_test₂,
                        chains[sim][chain].grammar) 
    ŷ_test₃ = evalmodel(chains[sim][chain].samples[sample],
                        x_test₃,
                        chains[sim][chain].grammar) 
    rmse_train[sample, chain, sim] = rmsd(y, ŷ_train)
    rmse_test₁[sample, chain, sim] = rmsd(y, ŷ_test₁)
    rmse_test₂[sample, chain, sim] = rmsd(y, ŷ_test₂)
    rmse_test₃[sample, chain, sim] = rmsd(y, ŷ_test₃)
end 
