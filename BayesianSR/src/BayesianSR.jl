module BayesianSR
export EqTree, Sample, Chain, nodetypes, ols, tableforeval, evaltree, optimβ!

using ExprRules, Distributions, Random, StatsBase, AbstractTrees

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
end 
Sample(k::Real, grammar::Grammar) = Sample([EqTree(grammar) for i in 1:k], zeros(k + 1))

struct Chain
    samples::Vector{Sample}
    grammar::Grammar
    x::Matrix{Float64}
    y::Vector{Float64}
    stats::Dict
    # ... hyperparams, statistics
    # TODO: I think a dictionary Prior with all hyperparameters and prios
end 

function Chain(x::Matrix, y::Vector, k::Int)
    grammar = append!(deepcopy(defaultgrammar), variablestogrammar(x))
    sample = [Sample(k, grammar)]
    stats = Dict([(:lastk, 0),
                  (:proposals, 0)])
    Chain(sample, grammar, x, y, stats)
end 

function Chain(x::Matrix, y::Vector, operators::Grammar, k::Int)
    grammar = append!(operators, variablestogrammar(x))
    sample = [Sample(k, grammar)]
    stats = Dict([(:lastk, 0),
                  (:proposals, 0)])
    Chain(sample, grammar, x, y, stats)
end 

include("grammars.jl")
include("evaltree.jl")
include("describetree.jl")
include("utils.jl")
include("ols.jl")
include("growtree.jl")
include("sampletree.jl")
# include("sampling.jl")
# include("mcmc.jl")

end # module
