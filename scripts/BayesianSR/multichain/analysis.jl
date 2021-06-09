using BayesianSR,
    Random,
    StatsBase,
    Distributions,
    JLD2,
    FileIO,
    ProgressMeter

filenames = readdir("./chains")
rx = r"chains-(\d+)-(\d+)"

no_sim = 0
no_exprs = 0

# Get total number of sims and chains
for file in filenames
    m = match(rx, file)
    global no_sim = max(no_sim, parse(Int, m.captures[1]))
    global no_exprs = max(no_exprs, parse(Int, m.captures[2]))
end 

# Load the first chain to get its dimensions and x
cs₀ = load("./chains/chains-1-1.jld2")["chains"]
no_chains = length(cs₀)
no_samples = length(cs₀[1])
x = cs₀[1].x
grammar = cs₀[1].grammar
n, m = size(x)
Random.seed!(1)
x_test₁ = rand(Uniform(-3 , 3), (n, m))
x_test₂ = rand(Uniform(-6 , 6), (n, m))
x_test₃ = rand(Uniform(3 , 6),  (n, m))

# 4D Arrays with dimensions samples/chain X no_chains X no_functions X no_simulations
rmse_train = Array{Float64}(undef, no_samples, no_chains, no_exprs, no_sim)
rmse_test₁ = Array{Float64}(undef, no_samples, no_chains, no_exprs, no_sim)
rmse_test₂ = Array{Float64}(undef, no_samples, no_chains, no_exprs, no_sim)
rmse_test₃ = Array{Float64}(undef, no_samples, no_chains, no_exprs, no_sim)

accepted = zeros(no_sim, no_exprs)

times = Matrix(undef, no_sim, no_exprs)

p = Progress(no_sim*no_exprs*no_samples*no_chains)
@time for sim_id=1:no_sim, expr_id=1:no_exprs
    l = load(string("./chains/chains-", sim_id, "-", expr_id, ".jld2"))
    cs = l["chains"]
    # Runtime of a multichain
    times[sim_id, expr_id] = l["t"]
    y = cs[1].y
    for chain_id in 1:no_chains
        # Accepted samples per multichain
        accepted[sim_id, expr_id] += cs[chain_id].stats[:accepted]

        for sample in 1:no_samples
            # evalmodel() with the training dataset will always converge
            # because samples that do not converge are not included in the chains,
            # but it might fail (e.g. sin(Inf)) with a testing dataset
            ŷ_train = evalmodel(cs[chain_id][sample], x, grammar) 
            try 
                global ŷ_test₁ = evalmodel(cs[chain_id][sample], x_test₁, grammar) 
            catch e 
                if isa(e, DomainError)
                    global  ŷ_test₁ = [Inf for _ in y]
                end 
            end 
            try 
                global ŷ_test₂ = evalmodel(cs[chain_id][sample], x_test₂, grammar) 
            catch e 
                if isa(e, DomainError)
                    global ŷ_test₂ = [Inf for _ in y]
                end 
            end 
            try 
                global ŷ_test₃ = evalmodel(cs[chain_id][sample], x_test₃, grammar) 
            catch e 
                if isa(e, DomainError)
                    global ŷ_test₃ = [Inf for _ in y]
                end 
            end 
            rmse_train[sample, chain_id, expr_id, sim_id] = rmsd(y, ŷ_train)
            rmse_test₁[sample, chain_id, expr_id, sim_id] = rmsd(y, ŷ_test₁)
            rmse_test₂[sample, chain_id, expr_id, sim_id] = rmsd(y, ŷ_test₂)
            rmse_test₃[sample, chain_id, expr_id, sim_id] = rmsd(y, ŷ_test₃)
            next!(p)
        end
    end 
end 

best_chain_train = Vector(undef, no_exprs)
best_chain_test₁ = Vector(undef, no_exprs)
best_chain_test₂ = Vector(undef, no_exprs)
best_chain_test₃ = Vector(undef, no_exprs)

for expr_id in 1:no_exprs
    best_chain_train[expr_id] = findmin(rmse_train[end, :, expr_id, 1])[2]
    best_chain_test₁[expr_id] = findmin(rmse_test₁[end, :, expr_id, 1])[2]
    best_chain_test₂[expr_id] = findmin(rmse_test₂[end, :, expr_id, 1])[2]
    best_chain_test₃[expr_id] = findmin(rmse_test₃[end, :, expr_id, 1])[2]
end 

rmse_train[end, best_chain_train, :, 1]
rmse_test₁[end, best_chain_test₁, :, 1]
rmse_test₂[end, best_chain_test₂, :, 1] 
rmse_test₃[end, best_chain_test₃, :, 1] 

best_rmse = Matrix(undef, 4, no_exprs)

for expr_id=1:no_exprs
    best_rmse[1, expr_id] = rmse_train[end, best_chain_train[expr_id], expr_id, 1]
    best_rmse[2, expr_id] = rmse_test₁[end, best_chain_test₁[expr_id], expr_id, 1]
    best_rmse[3, expr_id] = rmse_test₂[end, best_chain_test₂[expr_id], expr_id, 1]
    best_rmse[4, expr_id] = rmse_test₃[end, best_chain_test₃[expr_id], expr_id, 1]
end 

acceptance_ratios = accepted / no_samples / no_chains

eqs = Vector(undef, no_exprs)
for expr_id in 1:no_exprs
    cs = load(string("./chains/chains-", 1, "-", expr_id, ".jld2"))["chains"]
    eqs[expr_id] = get_function(cs[best_chain_train[expr_id]])
end 

jldsave("output.jld2"; times, acceptance_ratios, eqs, rmse_train, rmse_test₁, rmse_test₂, rmse_test₃)

