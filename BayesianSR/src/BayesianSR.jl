module BayesianSR
export EqTree, Sample, Chain, nodetypes, ols, tableforeval, evaltree, optimβ!

using ExprRules, Distributions, Random

struct EqTree
    S::RuleNode
    #   Θ::Thetas
end 

EqTree(grammar::Grammar) = EqTree(growtree!(grammar))

# TODO: Implement lt()
# struct Thetas
#     a::Vector{Float64}
#     b::Vector{Float64}
# end 

mutable struct Sample#{k}
    trees::Vector{EqTree}
    β::Vector{Float64}
end 
Sample(k::Real, grammar::Grammar) = Sample([EqTree(grammar) for i in 1:k], zeros(k+1))

struct Chain
    samples::Vector{Sample}
    operators::Grammar
    y::Vector{Float64}
    x::Matrix{Float64}
    # ... hyperparams, statistics
    # TODO: I think a dictionary Prior with all hyperparameters and prios
end 

include("grammars.jl")
include("evaltree.jl")
include("modifytree.jl")
include("growtree.jl")
include("describetree.jl")
include("utils.jl")
# include("sampling.jl")

end # module
