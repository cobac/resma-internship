using BayesianSR,
    Random,
    Distributions,
    JLD2,
    ProgressMeter

Random.seed!(3)
n = 30 # no. observations
m = 2 # no. features
k = 2 # no. trees per sample
x = rand(Uniform(-3 , 3), (n, m))
   
f₁ = x -> 2.5*x[1]^4 - 1.3*x[1]^3 + 0.5*x[2]^2 - 1.7*x[2]
f₂ = x -> 8*x[1]^2 + 8*x[2]^3 - 15
f₃ = x -> 0.2*x[1]^3 + 0.5*x[2]^3 - 1.2*x[2] - 0.5*x[1]
f₄ = x -> 1.5*exp(x[1]) + 5*cos(x[2])
f₅ = x -> 6 * sin(x[1]) * cos(x[2])
f₆ = x -> 1.35*x[1]*x[2] * 5.5*sin((x[1]-1)*(x[2]-1))

y₁ = f₁.(eachrow(x))
y₂ = f₂.(eachrow(x))
y₃ = f₃.(eachrow(x))
y₄ = f₄.(eachrow(x))
y₅ = f₅.(eachrow(x))
y₆ = f₆.(eachrow(x))
 
functions = @grammar begin
    Real = Real + Real
    Real = Real - Real
    Real = Real * Real 
    Real = Real / Real 
    Real = sin(Real) 
    Real = cos(Real) 
    Real = exp(Real)
    Real = Real^2
    Real = Real^3
end

# Each equation is a linear combination of k=2 symbolic trees
hyper = Hyperparams(k=2)

function mcmc!_sim(chains::Vector{Chain}, n_steps::Int; no_chains::Int, p_interchain_jump::AbstractFloat=0.05,
                   progress::Progress)
    for achain in chains resize!(achain.samples, 1 + n_steps) end
    for i in 2:(n_steps + 1)
        j = chains[1].stats[:lastj] + 1
        j == no_trees(chains[1]) + 1 ? j = 1 : nothing
        for chain_id in eachindex(chains)
            chains[chain_id].stats[:lastj] = j
            if rand() < p_interchain_jump
                base_chain = sample(deleteat!([1:no_chains;], chain_id))
                chains[chain_id].samples[i] = BayesianSR.step(chains[chain_id], chains[base_chain].samples[i-1], i - 1, j)
            else 
                chains[chain_id].samples[i] = BayesianSR.step(chains[chain_id], i - 1, j)
            end 
        end 
        next!(progress)
    end 
    return nothing
end 

function mcmc!_chunk(fun::Chain, no_iter::Int, no_chains::Int, p::Progress, sim_id::Int, expr_id::Int)
    no_iter == 1000 && no_iter/1000 % 1 !=0 &&
        error("Argument no_iter has to be a multiple of 1000 but it is ", no_iter)
    chains = push!([Chain([BayesianSR.new_sample_recursive(fun.grammar, fun.hyper, fun.x, fun.y)],
                          fun.grammar, fun.x, fun.y, deepcopy(fun.stats), fun.hyper)
                    for _ in 1:(no_chains - 1)], fun)
    for chunk in 1:(no_iter/1000)
        t = @elapsed mcmc!_sim(chains, 1000, no_chains = no_chains, progress = p)
        jldsave(string("./splitchains/chains-", sim_id, "-", expr_id, "-", Int(chunk), ".jld2"); chains, t)
        for chain in chains
            deleteat!(chain.samples, 1:(length(chain)-1))
        end 
    end 
    return nothing
end 

function simulation(no_sim, no_chains, no_iter)
    p = Progress(no_sim*no_iter*6)
    for sim_id in 1:no_sim
        expressions = [Chain(x, y₁, operators = functions, hyper = hyper),
                       Chain(x, y₂, operators = functions, hyper = hyper),
                       Chain(x, y₃, operators = functions, hyper = hyper),
                       Chain(x, y₄, operators = functions, hyper = hyper),
                       Chain(x, y₅, operators = functions, hyper = hyper),
                       Chain(x, y₆, operators = functions, hyper = hyper)]
        Threads.@threads for expr_id in eachindex(expressions)
            mcmc!_chunk(expressions[expr_id], no_iter, no_chains, p, sim_id, expr_id)
        end 
    end 
    return nothing
end 

@time simulation(10, 5, 100000)
