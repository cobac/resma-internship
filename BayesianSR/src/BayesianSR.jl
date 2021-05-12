module BayesianSR
using ExprRules, Distributions, Random, StatsBase, AbstractTrees, Parameters
import ExprRules: RuleNodeAndCount, RuleNode

export Chain, Hyperparams, step!, no_trees

"""
    Hyperparams(k = 3::Int, σ²_prior = InverseGamma(0.5, 0.5)::UnivariateDistribution)

Hyperparameters of a `Chain`.
"""
@with_kw struct Hyperparams
    k = 3::Int
    σ²_prior = InverseGamma(0.5, 0.5)::UnivariateDistribution
    σ²_a_prior = InverseGamma(0.5, 0.5)::UnivariateDistribution
    σ²_b_prior = InverseGamma(0.5, 0.5)::UnivariateDistribution
end 

"""
    RuleNode(grammar::Grammar, hyper::Hyperparams)

Samples a random `RuleNode` from the tree prior distribution with a `Grammar`.

See also: `growtree`
"""
ExprRules.RuleNode(grammar::Grammar, hyper::Hyperparams) = growtree(grammar, hyper)

"""
    LinearCoef(a::Float64, b::Float64)

Coefficients of a linear operator `a + b*x`.
"""
struct LinearCoef
    a::Float64
    b::Float64
end 

function LinearCoef(hyper::Hyperparams) 
    @unpack σ²_a_prior , σ²_b_prior = hyper
    σ²_a = rand(σ²_a_prior)
    σ²_b = rand(σ²_b_prior)
    a = rand(Normal(1, σ²_a))
    b = rand(Normal(1, σ²_b))
    return LinearCoef(a, b)
end 

function LinearCoef(σ²_a::AbstractFloat, σ²_b::AbstractFloat)
    a = rand(Normal(1, σ²_a))
    b = rand(Normal(1, σ²_b))
    return LinearCoef(a, b)
end 

"""
    Sample(trees::Vector{RuleNode}, β::Vector{Float64}, σ²::Float64)

Each sample of a `Chain` is one equation.

- `trees`: Vector with all `RuleNode`.
- `β`: Vector with the linear coefficients.
  - `β[1]`: Intercept.
- `σ²`: Variance of the residuals.

"""
mutable struct Sample
    trees::Vector{RuleNode}
    β::Vector{Float64}
    σ²::Float64
end 

"""
    Sample(k::Real, grammar::Grammar, prior::UnivariateDistribution)

- `k` is the number of `RuleNode` that are added in each equation.
- `prior` is the prior distribution of σ².
"""
function Sample(k::Real, grammar::Grammar, hyper::Hyperparams)
    @unpack σ²_prior = hyper
    Sample([RuleNode(grammar, hyper) for i in 1:k], zeros(k + 1), rand(σ²_prior))
end 

"""
    Chain

A chain of samples from the posterior space.

- `samples`: Vector will all the samples.
- `grammar`: The complete `Grammar` with all operators and features.
- `x`: Matrix of the features.
- `y`: Vector of the outcome values.
- `stats`: `Dict` with statistics about the `Chain`.
  - `:lastj`: Index of the last `RuleNode` that was sampled during MCMC.
- `hyper`: `Hyperparameters` of the `Chain`.

"""
struct Chain
    samples::Vector{Sample}
    grammar::Grammar
    x::Matrix{Float64}
    y::Vector{Float64}
    stats::Dict
    hyper::Hyperparams
end 

"""
    Chain(x::Matrix, y::Vector, operators::Grammar, hyper::Hyperparams)

Initialize a `Chain`.
"""
function Chain(x::Matrix, y::Vector, operators::Grammar, hyper::Hyperparams)
    @unpack k, σ²_prior = hyper
    grammar = append!(deepcopy(lineargrammar),
                      append!(deepcopy(operators), variablestogrammar(x)))
    sample = Sample(k, grammar, hyper)
    try 
        optimβ!(sample, x, y, grammar)
    catch e 
        sample.β = zeros(k+1)
    end 
    stats = Dict([(:lastj, 0)])
    return Chain([sample], grammar, x, y, stats, hyper)
end 

"""
    Chain(x::Matrix, y::Vector)

Initialize a `Chain` with default values.
"""
function Chain(x::Matrix, y::Vector)
    hyper = Hyperparams()
    operators = deepcopy(defaultgrammar)
    return Chain(x, y, operators, hyper)
end 

"""
    Chain(x::Matrix, y::Vector, k::Int)

Initialize a `Chain` with `k` `RuleNode` per `Sample`.
"""
function Chain(x::Matrix, y::Vector, k::Int)
    hyper = Hyperparams(k = k)
    operators = deepcopy(defaultgrammar)
    return Chain(x, y, operators, hyper)
end 

"""
    Chain(x::Matrix, y::Vector, hyper::Hyperparams)

Initialize a `Chain` with custom `Hyperparameters`.
"""
function Chain(x::Matrix, y::Vector, hyper::Hyperparams)
    operators = deepcopy(defaultgrammar)
    return Chain(x, y, operators, hyper)
end 

"""
    Chain(x::Matrix, y::Vector, grammar::Grammar)

Initialize a `Chain` with a custom `Grammar`.
"""
function Chain(x::Matrix, y::Vector, operators::Grammar)
    Chain(x, y, operators, Hyperparams())
end 


include("grammars.jl")
include("evaltree.jl")
include("utils.jl")
include("ols.jl")
include("growtree.jl")
include("treeprior.jl")
include("treemovements.jl")
include("sampling.jl")
include("mcmc.jl")

end # module
