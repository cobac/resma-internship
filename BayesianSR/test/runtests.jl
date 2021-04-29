using BayesianSR, Test

using ExprRules, Random, LinearAlgebra, Distributions, Parameters
#using ExprTools
#using AbstractTrees

Random.seed!(2)
n = 30
k = 3
β = rand(Uniform(-2, 2), k+1)
x = rand(Uniform(-10, 10), (n, k))
X = [ones(size(x)[1]) x]
ε = rand(Normal(0, 2), n)
y = X * β + ε

vargrammar = @grammar begin
    Real = x1 | x2 | x3
end 
fullgrammar = append!(deepcopy(BayesianSR.defaultgrammar), vargrammar)

@testset "Hyperparameters" begin
    hyper = Hyperparams()
    names = hyper |> typeof |> fieldnames
    @test  length(names) == 2
    @test :k in names
    @test :σ²_prior in names
    @unpack k, σ²_prior = hyper
    @test typeof(k) <: Int
    @test typeof(σ²_prior) <: UnivariateDistribution
    @test typeof(rand(σ²_prior)) <: AbstractFloat
end 

hyper = Hyperparams()

@testset "Grammars" begin
    var_operators = BayesianSR.nodetypes(vargrammar)
    @test length(var_operators) == 3
    @test in(0, var_operators)
    @test maximum(var_operators) == 0

    xgrammar = BayesianSR.variablestogrammar(x)
    @test vargrammar.bytype == xgrammar.bytype
    @test vargrammar.childtypes == xgrammar.childtypes
    @test vargrammar.iseval == xgrammar.iseval
    @test vargrammar.isterminal == xgrammar.isterminal
    @test vargrammar.rules == xgrammar.rules
    @test vargrammar.types == xgrammar.types

    default_operators = BayesianSR.nodetypes(BayesianSR.defaultgrammar)
    @test length(default_operators) == 6
    @test in(0, default_operators) == false
    @test in(1, default_operators)
    @test maximum(default_operators) == 2

    full_operators = BayesianSR.nodetypes(fullgrammar)
    @test length(full_operators) == length(var_operators) + length(default_operators)
    @test maximum(full_operators) == maximum(default_operators)
end


@testset "Tree generation" begin
    Random.seed!(5)
    tree = BayesianSR.EqTree(fullgrammar)
    table = BayesianSR.tableforeval(x, 3, fullgrammar)
    @test x[3, 1] == table[:x1]
    @test x[3, 2] == table[:x2]
    @test x[3, 3] == table[:x3]

    eq = get_executable(tree.S, fullgrammar)
    answ = Core.eval(table, eq)
    @test length(answ) == 1
    @test isreal(answ)
    @test answ ≈ 0.8910314500261972
    treex = BayesianSR.evaltree(tree, x, fullgrammar)
    @test length(treex) == size(x)[1]
end

@testset "Describe trees" begin
    tree = BayesianSR.EqTree(fullgrammar).S
    nodes = BayesianSR.flatten(tree)
    @test length(nodes) == BayesianSR.n_operators(tree, fullgrammar) +
        BayesianSR.n_terminals(tree, fullgrammar)
    root = RuleNode(3)
    @test BayesianSR.n_operators(root, fullgrammar) == 1
    @test BayesianSR.n_terminals(root, fullgrammar) == 0
    @test BayesianSR.n_candidates(root, fullgrammar) == 0
end 

@testset "Node sampling" begin
    tree = BayesianSR.EqTree(fullgrammar).S
    terminal = BayesianSR.sampleterminal(tree, fullgrammar)
    operator = BayesianSR.sampleoperator(tree, fullgrammar)
    terminal = get(tree, terminal).ind
    operator = get(tree, operator).ind
    @test nchildren(fullgrammar, terminal) == 0
    @test in(nchildren(fullgrammar, operator), [1,2])
end 

@testset "Samples" begin
    @unpack σ²_prior = hyper
    Random.seed!(3)
    k = 3
    model = BayesianSR.Sample(k, fullgrammar, σ²_prior)
    @test maximum(model.β) == 0
    @test length(model.β) == k+1
    @test length(model.trees) == k
    BayesianSR.optimβ!(model, x, y, fullgrammar)
    @test length(model.β) == k+1
    @test in(0, model.β) == false
end 

@testset "Chain initialization" begin
    k = 3
    chain = Chain(x, y)

end 

# TODO: Chain initialization
# TODO: Tree movements
# TODO: Tree sampling
# TODO: mcmc
