module BayesianSR
using ExprRules, Distributions, Random, StatsBase, AbstractTrees, Parameters
import ExprRules: RuleNodeAndCount

export Chain, Hyperparams

"""
    Hyperparams(k = 3::Int, σ²_prior = InverseGamma(0.5, 0.5)::UnivariateDistribution)

Hyperparameters of a `Chain`.
"""
@with_kw struct Hyperparams
    k = 3::Int
    σ²_prior = InverseGamma(0.5, 0.5)::UnivariateDistribution
end 

"""
    EqTree(S::RuleNode)

A symbolic tree.
"""
struct EqTree
    S::RuleNode
    #   Θ::Thetas
end 

"""
    EqTree(grammar::Grammar)

Samples a random `EqTree` from the prior distribution with a `Grammar`.

See also: `growtree`
"""
EqTree(grammar::Grammar) = EqTree(growtree(grammar))

# TODO: Implement lt()
# struct Thetas
#     a::Vector{Float64}
#     b::Vector{Float64}
# end 

"""
    Sample(trees::Vector{EqTree}, β::Vector{Float64}, σ²::Float64)

Each sample of a `Chain` is one equation.

- `trees`: Vector with all `EqTree`.
- `β`: Vector with the linear coefficients.
  - `β[1]`: Intercept.
- `σ²`: Variance of the residuals.

"""
mutable struct Sample
    trees::Vector{EqTree}
    β::Vector{Float64}
    σ²::Float64
end 

"""
    Sample(k::Real, grammar::Grammar, prior::UnivariateDistribution)

- `k` is the number of `EqTree` that are added in each equation.
- `prior` is the prior distribution of σ².
"""
function Sample(k::Real, grammar::Grammar, prior::UnivariateDistribution)
    Sample([EqTree(grammar) for i in 1:k], zeros(k + 1), rand(prior))
end 

"""
    Chain

A chain of samples from the posterior space.

- `samples`: Vector will all the samples.
- `grammar`: The complete `Grammar` with all operators and features.
- `x`: Matrix of the features.
- `y`: Vector of the outcome values.
- `stats`: `Dict` with statistics about the `Chain`.
  - `:lastj`: Index of the last `EqTree` that was sampled during MCMC.
  - `:proposals`: Number of proposed jumps in posterior space.
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
    grammar = append!(deepcopy(operators), variablestogrammar(x))
    sample = Sample(k, grammar, σ²_prior)
    optimβ!(sample, x, y, grammar)
    stats = Dict([(:lastj, 0),
                  (:proposals, 0)])
    Chain([sample], grammar, x, y, stats, hyper)
end 

"""
    Chain(x::Matrix, y::Vector)

Initialize a `Chain` with default values.
"""
function Chain(x::Matrix, y::Vector)
    hyper = Hyperparams()
    operators = deepcopy(defaultgrammar)
    Chain(x, y, operators, hyper)
end 

"""
    Chain(x::Matrix, y::Vector, k::Int)

Initialize a `Chain` with `k` `EqTree` per `Sample`.
"""
function Chain(x::Matrix, y::Vector, k::Int)
    hyper = Hyperparams(k = k)
    operators = deepcopy(defaultgrammar)
    Chain(x, y, operators, hyper)
end 

"""
    Chain(x::Matrix, y::Vector, hyper::Hyperparams)

Initialize a `Chain` with custom `Hyperparameters`.
"""
function Chain(x::Matrix, y::Vector, hyper::Hyperparams)
    operators = deepcopy(defaultgrammar)
    Chain(x, y, operators, hyper)
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
include("prior.jl")
include("sampletree.jl")
include("sampling.jl")
include("mcmc.jl")

end # module
