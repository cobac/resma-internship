using BayesianSR,
    ExprRules,
    Random,
    StatsBase,
    AbstractTrees,
    LinearAlgebra,
    Distributions,
    BenchmarkTools,
    GLMakie

Random.seed!(2)
n = 100 # no. observations
m = 2 # no. features
k = 2 # no. trees per sample
β = rand(Uniform(-2, 2), k+1)
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
 
functions = ExprRules.@grammar begin
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

chain₁ = Chain(x, y₁, operators = functions, hyper = hyper)
chain₂ = Chain(x, y₂, operators = functions, hyper = hyper)
chain₃ = Chain(x, y₃, operators = functions, hyper = hyper)
chain₄ = Chain(x, y₄, operators = functions, hyper = hyper)
chain₅ = Chain(x, y₅, operators = functions, hyper = hyper)
chain₆ = Chain(x, y₆, operators = functions, hyper = hyper)

function stepchains!(no_iter::Int)
    mcmc!(chain₁, no_iter)
    mcmc!(chain₂, no_iter)
    mcmc!(chain₃, no_iter, verbose = true)
    mcmc!(chain₄, no_iter)
    mcmc!(chain₅, no_iter)
    mcmc!(chain₆, no_iter)
end 

@time stepchains!(1000)
# Seems to be way better convergence with 10k samples
# ~8ms per sample

chain₁.stats[:accepted] # / length(chain₁) 
chain₂.stats[:accepted] # / length(chain₂) 
chain₃.stats[:accepted] # / length(chain₃) 
chain₄.stats[:accepted] # / length(chain₄) 
chain₅.stats[:accepted] # / length(chain₅) 
chain₆.stats[:accepted] # / length(chain₆) 
# All extremely low :(

function geteq(chain)
        trees = chain.samples[end].trees
        eqs = [get_executable(tree, chain.grammar) for tree in trees]
    β = chain.samples[end].β
    eq = :($(β[1]) + $(β[2]) * $(eqs[1]) + $(β[3]) * $(eqs[2]))
end 

eq₁ = geteq(chain₁)
# Gets the exponents and coeffs except x2^2
# Without noise is worse,
# but graphically is excellent
eq₂ = geteq(chain₂)
# Gets the exponents and ~kinda the coefficients
# Without noise it's recovered perfectly
eq₃ = geteq(chain₃)
# Something is not working, NaNs, not accepting samples
eq₄ = geteq(chain₄)
# Eh, gets exp(x1) but.. no
# Without noise it's recovered perfectly
eq₅ = geteq(chain₅)
# Nope..
# Nope without noise either
# However, numerically is close (see rmsd)
# Graphically is very close
eq₆ = geteq(chain₆)
# Nope..
# Nope without noise either
# Graphically is very close

function dirtyeval(f, x)
    out = Vector{Float64}(undef, size(x, 1))
    symboltable = SymbolTable(chain₁.grammar)
    symboltable[:linear_operator] = BayesianSR.linear_operator
    for i in axes(x, 1)
        symboltable[:x1] = x[i, 1]
        symboltable[:x2] = x[i, 2]
        out[i] = Core.eval(symboltable, f)
    end 
    return out
end

ŷ₁ = dirtyeval(eq₁, x)
ŷ₂ = dirtyeval(eq₂, x)
# ŷ₃ = dirtyeval(eq₃, x)
ŷ₄ = dirtyeval(eq₄, x)
ŷ₅ = dirtyeval(eq₅, x)
ŷ₆ = dirtyeval(eq₆, x)

rmsd(y₁, ŷ₁)
rmsd(y₂, ŷ₂)
# rmsd(y₃, ŷ₃)
rmsd(y₄, ŷ₄)
rmsd(y₅, ŷ₅)
rmsd(y₆, ŷ₆)

scatter(x[: , 1], ŷ₁)
scatter!(x[: , 1], y₁, color = :gray)
scatter(x[: , 2], ŷ₁)
scatter!(x[: , 2], y₁, color = :gray)

scatter(x[: , 1], ŷ₂)
scatter!(x[: , 1], y₂, color = :gray)
scatter(x[: , 2], ŷ₂)
scatter!(x[: , 2], y₂, color = :gray)

scatter(x[: , 1], ŷ₄)
scatter!(x[: , 1], y₄, color = :gray)
scatter(x[: , 2], ŷ₄)
scatter!(x[: , 2], y₄, color = :gray)

scatter(x[: , 1], ŷ₅)
scatter!(x[: , 1], y₅, color = :gray)
scatter(x[: , 2], ŷ₅)
scatter!(x[: , 2], y₅, color = :gray)

scatter(x[: , 1], ŷ₆)
scatter!(x[: , 1], y₆, color = :gray)
scatter(x[: , 2], ŷ₆)
scatter!(x[: , 2], y₆, color = :gray)
