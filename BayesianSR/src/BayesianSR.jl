module BayesianSR
export EqTree, Sample, Chain, operatortypes, ols, tableforeval, evaltree, optimβ!

using ExprRules
# using AbstractTrees
using Distributions
using Random

mutable struct EqTree
    S::RuleNode
    #   Θ::Thetas
end 

EqTree(grammar::Grammar) = EqTree(rand(RuleNode, grammar, :Real, 20))

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

mutable struct Chain
    samples::Vector{Sample}
    operators::Grammar
    y::Vector{Float64}
    x::Matrix{Float64}
    # ... hyperparams, statistics
end 

include("grammars.jl")
include("evaltree.jl")
include("modifytree.jl")

#------------------------ To move:
function operatortypes(grammar=defaultgrammar)
    #= Classifies operators into unary-binary-terminal =#
    types = [ExprRules.nchildren(grammar, i)
             for i in 1:length(grammar.rules)]
    return types
end 

end # module
