using BayesianSR,
    Random,
    StatsBase,
    Distributions,
    JLD2

chains = []
t = []

for file in readdir("./chains")
    l = load(string("./chains/", file))
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
    global accepted₁ +=  chains[i][1].stats[:accepted]
    global accepted₂ +=  chains[i][2].stats[:accepted]
    global accepted₃ +=  chains[i][3].stats[:accepted]
    global accepted₄ +=  chains[i][4].stats[:accepted]
    global accepted₅ +=  chains[i][5].stats[:accepted]
    global accepted₆ +=  chains[i][6].stats[:accepted]
end 

acceptance_ratios = Vector(undef, 6)
acceptance_ratios[1] = accepted₁ / length(chains[1][1]) / length(chains)
acceptance_ratios[2] = accepted₂ / length(chains[1][2]) / length(chains)
acceptance_ratios[3] = accepted₃ / length(chains[1][3]) / length(chains)
acceptance_ratios[4] = accepted₄ / length(chains[1][4]) / length(chains)
acceptance_ratios[5] = accepted₅ / length(chains[1][5]) / length(chains)
acceptance_ratios[6] = accepted₆ / length(chains[1][6]) / length(chains)

eqs = Vector(undef, 6)
eqs[1] = get_function(chains[1][1], latex = false)
eqs[2] = get_function(chains[2][2], latex = false)
eqs[3] = get_function(chains[3][3], latex = false)
eqs[4] = get_function(chains[4][4], latex = false)
eqs[5] = get_function(chains[5][5], latex = false)
eqs[6] = get_function(chains[6][6], latex = false)

# 3D Array with dimensions samples/chain X no_functions X no_simulations
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
    fun=1:length(chains[1]),
    sim=1:length(chains)
    y = chains[sim][fun].y
    # evalmodel() with the training dataset will always converge
    # because samples that do not converge are not included in the chains,
    # but it might fail (e.g. sin(Inf)) with a testing dataset
    ŷ_train = evalmodel(chains[sim][fun].samples[sample], x, chains[sim][fun].grammar) 
    try 
        global ŷ_test₁ = evalmodel(chains[sim][fun].samples[sample], x_test₁, chains[sim][fun].grammar) 
    catch e 
        if isa(e, DomainError)
            global ŷ_test₁ = [Inf for _ in y]
        end 
    end 
    try 
        global ŷ_test₂ = evalmodel(chains[sim][fun].samples[sample], x_test₂, chains[sim][fun].grammar) 
    catch e 
        if isa(e, DomainError)
            global ŷ_test₂ = [Inf for _ in y]
        end 
    end 
    try 
        global ŷ_test₃ = evalmodel(chains[sim][fun].samples[sample], x_test₃, chains[sim][fun].grammar) 
    catch e 
        if isa(e, DomainError)
            global ŷ_test₃ = [Inf for _ in y]
        end 
    end 
    rmse_train[sample, fun, sim] = rmsd(y, ŷ_train)
    rmse_test₁[sample, fun, sim] = rmsd(y, ŷ_test₁)
    rmse_test₂[sample, fun, sim] = rmsd(y, ŷ_test₂)
    rmse_test₃[sample, fun, sim] = rmsd(y, ŷ_test₃)
end 

# Expected: false
isnan.(rmse_test₁) |> any
isnan.(rmse_test₂) |> any
isnan.(rmse_test₃) |> any

# Expected: true
isinf.(rmse_test₁) |> any
isinf.(rmse_test₂) |> any
isinf.(rmse_test₃) |> any

jldsave("output.jld2"; t, acceptance_ratios, eqs, rmse_train, rmse_test₁, rmse_test₂, rmse_test₃)
