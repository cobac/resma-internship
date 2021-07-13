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
   
f₁(x) = 2.5*x[1]^4 - 1.3*x[1]^3 + 0.5*x[2]^2 - 1.7*x[2]
f₂(x) = 8*x[1]^2 + 8*x[2]^3 - 15
f₃(x) = 0.2*x[1]^3 + 0.5*x[2]^3 - 1.2*x[2] - 0.5*x[1]
f₄(x) = 1.5*exp(x[1]) + 5*cos(x[2])
f₅(x) = 6 * sin(x[1]) * cos(x[2])
f₆(x) = 1.35*x[1]*x[2] * 5.5*sin((x[1]-1)*(x[2]-1))

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

function mcmc!_sim(chain::Chain, no_iter::Int, p::Progress, sim_id::Int, chain_id::Int)
    no_iter == 1000 && no_iter/1000 % 1 !=0 && error("No_iter has to be a multiple of 1000 but it is ", no_iter)
    for chunk in 1:(no_iter/1000)
        t = @elapsed mcmc!(chain, 1000, progress = p, verbose = false)
        jldsave(string("./splitchains/chains-", sim_id, "-", chain_id, "-", Int(chunk),  ".jld2"); chain, t)
        deleteat!(chain.samples, 1:(length(chain)-1))
    end 
    return nothing
end 

function stepchains!(chains::Vector{Chain}, no_iter::Int, p::Progress, sim_id::Int)
    for chain in eachindex(chains)
        mcmc!_sim(chains[chain], no_iter, p, sim_id, chain)
    end 
end 

function simulation(no_sim, no_iter)
    p = Progress(no_sim*no_iter*6)
    Threads.@threads for sim_id in 1:no_sim
        chains = [Chain(x, y₁, operators = functions, hyper = hyper),
                  Chain(x, y₂, operators = functions, hyper = hyper),
                  Chain(x, y₃, operators = functions, hyper = hyper),
                  Chain(x, y₄, operators = functions, hyper = hyper),
                  Chain(x, y₅, operators = functions, hyper = hyper),
                  Chain(x, y₆, operators = functions, hyper = hyper)]
        stepchains!(chains, no_iter, p, sim_id)
    end 
end 

@time simulation(50, 100000)
