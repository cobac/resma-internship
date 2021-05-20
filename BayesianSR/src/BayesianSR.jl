module BayesianSR
using ExprRules, Distributions, Random, StatsBase, Parameters
import ExprRules: RuleNodeAndCount, RuleNode

export Chain, Hyperparams, mcmc!, no_trees

"""
    Hyperparams(k = 3::Int, σ²_prior = InverseGamma(0.5, 0.5)::UnivariateDistribution)

Hyperparameters of a `Chain`.
"""
@with_kw struct Hyperparams
    k = 3::Int
    σ²_prior::UnivariateDistribution = InverseGamma(0.5, 0.5)
    σ²_a_prior::UnivariateDistribution = InverseGamma(0.5, 0.5)
    σ²_b_prior::UnivariateDistribution = InverseGamma(0.5, 0.5)
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
    b = rand(Normal(0, σ²_b))
    return LinearCoef(a, b)
end 

function LinearCoef(σ²_a::AbstractFloat, σ²_b::AbstractFloat)
    a = rand(Normal(1, σ²_a))
    b = rand(Normal(0, σ²_b))
    return LinearCoef(a, b)
end 

"""
    Sample(trees::Vector{RuleNode}, β::Vector{Float64}, σ²::Float64)

Each sample of a `Chain` is one equation.

- `trees`: Vector with all `RuleNode`.
- `β`: Vector with the linear coefficients.
  - `β[1]`: Intercept.
- `σ²`: Dictionary with variances
  - `:σ²`: of the residuals
  - `:σ²_a`: of the LinearCoef intercepts
  - `:σ²_b`: of the LinearCoef slopes

"""
struct Sample
    trees::Vector{RuleNode}
    β::Vector{Float64}
    σ²::Dict{Symbol, AbstractFloat}
end 

"""
    Sample(k::Real, grammar::Grammar, hyper::Hyperparams)

- `k` is the number of `RuleNode` that are added in each equation.
"""
function Sample(k::Real, grammar::Grammar, hyper::Hyperparams)
    @unpack σ²_prior, σ²_a_prior, σ²_b_prior = hyper
    Sample([RuleNode(grammar, hyper) for _ in 1:k],
           zeros(k + 1),
           Dict([(:σ², rand(σ²_prior)),
                 (:σ²_a, 0),
                 (:σ²_b, 0)]))
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
    stats::Dict{Symbol, Int}
    hyper::Hyperparams
end 

"""
    Chain(x::Matrix, y::Vector; operators::Grammar = deepcopy(defaultgrammar), hyper::Hyperparams = Hyperparams())

Initialize a `Chain`.
"""
function Chain(x::Matrix, y::Vector;
               operators::Grammar = deepcopy(defaultgrammar),
               hyper::Hyperparams = Hyperparams())
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

include("grammars.jl")
include("evaltree.jl")
include("utils.jl")
include("ols.jl")
include("growtree.jl")
include("treeprior.jl")
include("treemovements.jl")
include("treeproposal.jl")
include("coefproposal.jl")
include("mcmc.jl")

end # module
