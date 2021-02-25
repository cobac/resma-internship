module BayesianSR
export EqTree, Sample, fullgrammar, Chain, operatortypes, ols, tableforeval, evaltree, optimβ!

using ExprRules
# using AbstractTrees
using Distributions
using Random

mutable struct EqTree
    S::RuleNode
    #   Θ::Thetas
end 

EqTree() = EqTree(rand(RuleNode, fullgrammar, :Real, 20))

# TODO: Implement lt()
# struct Thetas
#     a::Vector{Float64}
#     b::Vector{Float64}
# end 

mutable struct Sample#{k}
    trees::Vector{EqTree}
    β::Vector{Float64}
end 
Sample(k) = Sample([EqTree() for i in 1:k], zeros(k+1))

mutable struct Chain
    samples::Vector{Sample}
    operators::Grammar
    y::Vector{Float64}
    x::Matrix{Float64}
    # ... hyperparams, statistics
end 

const defaultgrammar = ExprRules.@grammar begin
    Real = Real + Real
    Real = Real - Real
    Real = Real * Real 
    Real = Real / Real
    Real = cos(Real) 
    Real = sin(Real) 
end

# TODO: Ask for custom grammar

# TODO(Don): Generate variables grammar from variables
const variablesgrammar = ExprRules.@grammar begin
    Real = x1 | x2 | x3
end

function Base.append!(grammar1::Grammar, grammar2::Grammar)
    N = length(grammar1.rules)
    append!(grammar1.rules, grammar2.rules)
    append!(grammar1.types, grammar2.types)
    append!(grammar1.isterminal, grammar2.isterminal)
    append!(grammar1.iseval, grammar2.iseval)
    append!(grammar1.childtypes, copy.(grammar2.childtypes))
    for (s, v) in grammar2.bytype
        grammar1.bytype[s] = append!(get(grammar1.bytype, s, Int[]), N .+ v)
    end
    grammar1
end

# TODO(Don): How to import it?
# fullgrammar = ExprRules.Base.append!(deepcopy(defaultgrammar), variablesgrammar)
const fullgrammar = append!(deepcopy(defaultgrammar), variablesgrammar)

function operatortypes(grammar=defaultgrammar)
    #= Classifies operators into unary-binary-terminal =#
    types = [ExprRules.nchildren(grammar, i)
             for i in 1:length(grammar.rules)]
    return types
end 

function evaltree(tree, x)
    #= Evaluates a tree into a vector for every variable x_j =#
    n = size(x)[1]
    out = Vector{Float64}(undef, n)
    for i in 1:n
        table = tableforeval(x, i, fullgrammar) 
        eq = ExprRules.get_executable(tree.S, fullgrammar)
        out[i] = Core.eval(table, eq)
    end 
    return out
end 

function tableforeval(x, i, grammar=fullgrammar)
    #= Generates a symbol table ready for evaluation for x_i =#
    symboltable = ExprRules.SymbolTable(grammar)
    k = size(x)[2]
    for m in 1:k
        symboltable[Symbol("x", m)] = x[i, m]
    end 
    return symboltable
end 

function optimβ!(model, y, x)
    #= Minimizes model.β by OLS =#
    n = size(x)[1]
    xs = Matrix{Float64}(undef, (n, length(model.trees)))
    for k in 1:length(model.trees)
        xs[:, k] = evaltree(model.trees[k], x)
    end 
    model.β = ols(y, xs)
    return nothing
end 

function ols(y, x)
    X = [ones(size(x)[1]) x]
    β = inv(X' * X) * X' * y
    return β
end 

end # module
