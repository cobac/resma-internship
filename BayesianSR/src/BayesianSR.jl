module BayesianSR
using ExprRules, Distributions, Random, StatsBase, AbstractTrees, Parameters
import ExprRules: RuleNodeAndCount

export Chain, Hyperparams

@with_kw struct Hyperparams
    k = 3::Int
    σ²_prior = InverseGamma(0.5, 0.5)::UnivariateDistribution
end 

struct EqTree
    S::RuleNode
    #   Θ::Thetas
end 

EqTree(grammar::Grammar) = EqTree(growtree(grammar))

# TODO: Implement lt()
# struct Thetas
#     a::Vector{Float64}
#     b::Vector{Float64}
# end 

mutable struct Sample
    trees::Vector{EqTree}
    β::Vector{Float64}
    σ²::Float64
end 

function Sample(k::Real, grammar::Grammar, prior::UnivariateDistribution)
    Sample([EqTree(grammar) for i in 1:k], zeros(k + 1), rand(prior))
end 

struct Chain
    samples::Vector{Sample}
    grammar::Grammar
    x::Matrix{Float64}
    y::Vector{Float64}
    stats::Dict
    hyper::Hyperparams
end 

function Chain(x::Matrix, y::Vector)
    hyper = Hyperparams()
    grammar = deepcopy(defaultgrammar)
    Chain(x, y, grammar, hyper)
end 

function Chain(x::Matrix, y::Vector, k::Int)
    hyper = Hyperparams(k = k)
    grammar = deepcopy(defaultgrammar)
    Chain(x, y, grammar, hyper)
end 

function Chain(x::Matrix, y::Vector, hyper::Hyperparams)
    grammar = deepcopy(defaultgrammar)
    Chain(x, y, grammar, hyper)
end 

function Chain(x::Matrix, y::Vector, grammar::Grammar)
    Chain(x, y, grammar, Hyperparams())
end 

function Chain(x::Matrix, y::Vector, operators::Grammar, hyper::Hyperparams)
    @unpack k, σ²_prior = hyper
    grammar = append!(deepcopy(operators), variablestogrammar(x))
    sample = Sample(k, grammar, σ²_prior)
    optimβ!(sample, x, y, grammar)
    stats = Dict([(:lastj, 0),
                  (:proposals, 0)])
    Chain([sample], grammar, x, y, stats, hyper)
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
