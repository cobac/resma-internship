include("BayesianSR.jl")
using .BayesianSR

using ExprRules
using Random
#using ExprTools
using AbstractTrees
using LinearAlgebra
using Distributions

begin 
    n = 30
    k = 3
    β = rand(Uniform(-2, 2), k+1)
    x = rand(Uniform(-10, 10), (n, k))
    X = [ones(size(x)[1]) x]
    ε = rand(Normal(0, 2), n)
    y = X * β + ε
end 

begin
    operators = operatortypes(BayesianSR.fullgrammar)
    symbol_table = SymbolTable(BayesianSR.fullgrammar)
    Random.seed!(2)
    tree = BayesianSR.EqTree()
    table = BayesianSR.tableforeval(x, 3)
    eq = get_executable(tree.S, BayesianSR.fullgrammar)
    answ = Core.eval(table, eq)
    treex = BayesianSR.evaltree(tree, x)
end 

begin
    g1 = BayesianSR.defaultgrammar
    g2 = BayesianSR.variablesgrammar
    g3 = BayesianSR.fullgrammar
end 

begin
    Random.seed!(3)
    model = BayesianSR.Sample(3)
    BayesianSR.optimβ!(model, y, x)
end 


# Modify trees
#equation = get_executable(tree, grammar)
#print_tree(equation)

#node_index = sample(NodeLoc, tree)
#old_node = get(tree, node_index)
#new_node = rand(RuleNode, grammar, :Real, 3)
#insert!(tree, node_index, new_node)
#new_equation = get_executable(tree, grammar)



