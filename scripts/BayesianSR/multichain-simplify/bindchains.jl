using BayesianSR, JLD2, FileIO, ProgressMeter

filenames = readdir("./splitchains")
rx = r"chains-(\d+)-(\d+)-(\d+)"

no_sim = 0
no_exprs = 0
no_chunks = 0

# Get total number of sims, chains and chunks
for file in filenames
    m = match(rx, file)
    global no_sim = max(no_sim, parse(Int, m.captures[1]))
    global no_exprs = max(no_exprs, parse(Int, m.captures[2]))
    global no_chunks = max(no_chunks, parse(Int, m.captures[3]))
end 

# Bind all chunks of the same chain together
function Base.append!(chain1::Chain, chain2::Chain)
    append!(chain1.samples, chain2.samples[2:end])
    return chain1
end 

p = Progress(no_sim*no_exprs*no_chunks)
@time for sim=1:no_sim, expr_id=1:no_exprs
    file = string("./splitchains/chains-", sim, "-", expr_id, "-", 1, ".jld2")
    l = load(file)
    chains = l["chains"]
    t = l["t"]
    rm(file)
    for chunk in 2:no_chunks
        file = string("./splitchains/chains-", sim, "-", expr_id, "-", chunk, ".jld2")
        l = load(file)
        rm(file)
        for chain_id in eachindex(chains)
            append!(chains[chain_id], l["chains"][chain_id])
            t += l["t"]
        end 
        next!(p)
    end 
    jldsave(string("./chains/chains-", sim, "-", expr_id, ".jld2"); chains, t)
end 
