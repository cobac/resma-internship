using BayesianSR,
    ExprRules,
    Random,
    StatsBase,
    AbstractTrees,
    LinearAlgebra,
    Distributions,
    BenchmarkTools

Random.seed!(2)
n = 100 # no. observations
m = 2 # no. features
k = 2 # no. trees per sample
β = rand(Uniform(-2, 2), k+1)
x = rand(Uniform(-3 , 3), (n, m))
   
# f₁ = x -> 2.5*x[1]^4 - 1.3*x[1]^3 + 0.5*x[2]^2 - 1.7*x[2]
# f₂ = x -> 8*x[1]^2 + 8*x[2]^3 - 15
# f₃ = x -> 0.2*x[1]^3 + 0.5*x[2]^3 - 1.2*x[2] - 0.5*x[1]
# f₄ = x -> 1.5*exp(x[1]) + 5*cos(x[2])
# f₅ = x -> 6 * sin(x[1]) * cos(x[2])
# f₆ = x -> 1.35*x[1]*x[2] * 5.5*sin((x[1]-1)(x[2]-1))

f₁ = x -> x[1]^4 - x[1]^3 + x[2]^2 - x[2]
f₄ = x -> exp(x[1]) + cos(x[2])
f₅ = x -> 6 * sin(x[1]) * cos(x[2])
f₆ = x -> x[1]*x[2] * sin(x[1]*x[2])

function applyf(f, x)
    out = Vector{Float64}(undef, size(x)[1])
    for i in 1:size(x)[1]
        out[i] = f(x[i, :])
    end 
    return out
end  
 
y₁ = applyf(f₁, x) .+ rand(Normal(0, 2))
y₄ = applyf(f₄, x) .+ rand(Normal(0, 2))
y₅ = applyf(f₅, x) .+ rand(Normal(0, 2))
y₆ = applyf(f₆, x) .+ rand(Normal(0, 2))
 
 
operators = ExprRules.@grammar begin
       Real = Real + Real
       Real = Real * Real 
       Real = cos(Real) 
       Real = sin(Real) 
       Real = exp(Real)
       Real = Real^2
       Real = Real^3
   end

hyper = Hyperparams(k = 2)

chain₁ = Chain(x, y₁, operators, hyper )
chain₄ = Chain(x, y₄, operators, hyper )
chain₅ = Chain(x, y₅, operators, hyper )
chain₆ = Chain(x, y₆, operators, hyper )

chain₁
chain₁.samples[1].trees
chain₁.samples[1].β
chain₁.samples[1].σ²
chain₁.grammar
chain₁.stats
chain₁.hyper


# @btime step!(chain₁) # 5ms

no_iter = 1000
for i in 1:no_iter
    i%%100 == 0 && println("Iteration no. ", i)
    step!(chain₁)
    step!(chain₄)
    step!(chain₅)
    step!(chain₆)
end 

length(chain₁) / chain₁.stats[:proposals] # 12%
length(chain₄) / chain₄.stats[:proposals] # 30%
length(chain₅) / chain₅.stats[:proposals] # 31%
length(chain₆) / chain₆.stats[:proposals] # 25%

function geteq(chain)
    trees = chain.samples[end].trees
    eqs = [get_executable(tree.S, chain.grammar) for tree in trees]
    β = chain.samples[end].β
    eq = :($(β[1]) + $(β[2]) * $(eqs[1]) + $(β[3]) * $(eqs[2]))
end 

eq₁ = geteq(chain₁)
eq₄ = geteq(chain₄)
eq₅ = geteq(chain₅)
eq₆ = geteq(chain₆)
