using BayesianSR,
    Random,
    StatsBase,
    Distributions,
    JLD2,
    FileIO,
    ProgressMeter

filenames = readdir("./chains")
rx = r"chain-(\d+)-(\d+)"

no_sim = 0
no_chains = 0

# Get total number of sims and chains
for file in filenames
    m = match(rx, file)
    global no_sim = max(no_sim, parse(Int, m.captures[1]))
    global no_chains = max(no_chains, parse(Int, m.captures[2]))
end 

# Load the first chain to get its dimensions and x
c₀ = load("./chains/chain-1-1.jld2")["chain"]
no_samples = length(c₀)
x = c₀.x
n, m = size(x)
Random.seed!(1)
x_test₁ = rand(Uniform(-3 , 3), (n, m))
x_test₂ = rand(Uniform(-6 , 6), (n, m))
x_test₃ = rand(Uniform(3 , 6),  (n, m))

# 3D Arrays with dimensions samples/chain X no_functions X no_simulations
rmse_train = Array{Float64}(undef, no_samples, no_chains, no_sim)
rmse_test₁ = Array{Float64}(undef, no_samples, no_chains, no_sim)
rmse_test₂ = Array{Float64}(undef, no_samples, no_chains, no_sim)
rmse_test₃ = Array{Float64}(undef, no_samples, no_chains, no_sim)

accepted = Matrix(undef, no_sim, no_chains)
eqs = Vector(undef, no_chains)

times = Matrix(undef, no_sim, no_chains)

p = Progress(no_sim*no_chains*no_samples)
for sim_id=1:no_sim, fun=1:no_chains
    l = load(string("./chains/chain-", sim_id, "-", fun, ".jld2"))
    c = l["chain"]
    # Runtime of a chain
    times[sim_id, fun] = l["t"]
    y = c.y
    # Accepted samples
    accepted[sim_id, fun] = c.stats[:accepted]
    # Example mathematical expressions
    sim_id == 1 ? eqs[fun] = get_function(c) : nothing

    for sample in 1:no_samples
        # evalmodel() with the training dataset will always converge
        # because samples that do not converge are not included in the chains,
        # but it might fail (e.g. sin(Inf)) with a testing dataset
        ŷ_train = evalmodel(c[sample], x, c.grammar) 
        try 
            global ŷ_test₁ = evalmodel(c[sample], x_test₁, c.grammar) 
        catch e 
            if isa(e, DomainError)
                global  ŷ_test₁ = [Inf for _ in y]
            end 
        end 
        try 
            global ŷ_test₂ = evalmodel(c[sample], x_test₂, c.grammar) 
        catch e 
            if isa(e, DomainError)
                global ŷ_test₂ = [Inf for _ in y]
            end 
        end 
        try 
            global ŷ_test₃ = evalmodel(c[sample], x_test₃, c.grammar) 
        catch e 
            if isa(e, DomainError)
                global ŷ_test₃ = [Inf for _ in y]
            end 
        end 
        rmse_train[sample, fun, sim_id] = rmsd(y, ŷ_train)
        rmse_test₁[sample, fun, sim_id] = rmsd(y, ŷ_test₁)
        rmse_test₂[sample, fun, sim_id] = rmsd(y, ŷ_test₂)
        rmse_test₃[sample, fun, sim_id] = rmsd(y, ŷ_test₃)
        next!(p)
    end 
end 

rmse_train[end, :, 1]
rmse_test₁[end, :, 1]
rmse_test₂[end, :, 1] 
rmse_test₃[end, :, 1] 

acceptance_ratios = accepted / no_samples

jldsave("output.jld2"; times, acceptance_ratios, eqs, rmse_train, rmse_test₁, rmse_test₂, rmse_test₃)

