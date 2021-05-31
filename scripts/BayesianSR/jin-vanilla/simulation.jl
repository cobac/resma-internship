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

function applyf(f, x)
    out = Vector{Float64}(undef, size(x, 1))
    for i in axes(x, 1)
        out[i] = f(x[i, :])
    end 
    return out
end  

y₁ = applyf(f₁, x) # .+ rand(Normal(0, 2))
y₂ = applyf(f₂, x) # .+ rand(Normal(0, 2))
y₃ = applyf(f₃, x) # .+ rand(Normal(0, 2))
y₄ = applyf(f₄, x) # .+ rand(Normal(0, 2))
y₅ = applyf(f₅, x) # .+ rand(Normal(0, 2))
y₆ = applyf(f₆, x) # .+ rand(Normal(0, 2))
 
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

function stepchains!(chains::Vector{Chain}, no_iter::Int)
    t = 0
    for chain in eachindex(chains)
        t += @elapsed mcmc!(chains[chain], no_iter, verbose = false)
    end 
    return t
end 

function simulation(no_sim, no_iter)
    p = Progress(no_sim)
    Threads.@threads for i in 1:no_sim
        chains = [Chain(x, y₁, operators = functions, hyper = hyper),
                  Chain(x, y₂, operators = functions, hyper = hyper),
                  Chain(x, y₃, operators = functions, hyper = hyper),
                  Chain(x, y₄, operators = functions, hyper = hyper),
                  Chain(x, y₅, operators = functions, hyper = hyper),
                  Chain(x, y₆, operators = functions, hyper = hyper)]
        t = stepchains!(chains, no_iter)
        save(string("./chains/chains", i, ".jld2"), "chains", chains, "t", t)
        next!(p)
    end 
end 

@time simulation(50, 100)
