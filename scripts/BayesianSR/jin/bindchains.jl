using BayesianSR, JLD2, FileIO


filenames = readdir("./splitchains")
rx = r"chains-(\d+)-(\d+)-(\d+)"

no_sim = 0
no_chains = 0
no_chunks = 0

# Get total number of sims, chains and chunks
for file in filenames
    m = match(rx, file)
    global no_sim = max(no_sim, parse(Int, m.captures[1]))
    global no_chains = max(no_chains, parse(Int, m.captures[2]))
    global no_chunks = max(no_chunks, parse(Int, m.captures[3]))
end 

# Bind all chunks of the same chain together
function Base.append!(chain1::Chain, chain2::Chain)
    append!(chain1.samples, chain2.samples[2:end])
    return chain1
end 

for sim=1:no_sim, chain_id=1:no_chains
    l = load(string("./splitchains/chains-", sim, "-", chain_id, "-", 1, ".jld2"))
    chain = l["chain"]
    t = l["t"]
    for chunk in 2:no_chunks
        l = load(string("./splitchains/chains-", sim, "-", chain_id, "-", chunk, ".jld2"))
        append!(chain, l["chain"])
        t += l["t"]
    end 
    jldsave(string("./chains/chain-", sim, "-", chain_id, ".jld2"); chain, t)
end 
