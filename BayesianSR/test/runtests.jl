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

function test_hyperparams(hyper::Hyperparams)
    @testset "Hyperparameters" begin
        names = hyper |> typeof |> fieldnames
        @test  length(names) == 2
        @test :k in names
        @test :σ²_prior in names
        @unpack k, σ²_prior = hyper
        @test typeof(k) <: Int
        @test typeof(σ²_prior) <: UnivariateDistribution
        @test typeof(rand(σ²_prior)) <: AbstractFloat
    end 
end 

hyper = Hyperparams()
test_hyperparams(hyper)

@testset "Grammars utils" begin
    node_types = BayesianSR.nodetypes(fullgrammar)
    @test length(node_types) == length(fullgrammar)
    operator_is = findall(x -> x==1 || x==2, node_types)
    terminal_is = findall(x -> x==0, node_types)
    @test length(operator_is) + length(terminal_is) == length(node_types)
    @test BayesianSR.operator_indices(fullgrammar) == operator_is
    @test BayesianSR.terminal_indices(fullgrammar) == terminal_is
end 

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

function test_tree(tree::RuleNode)
    @testset "Node sampling" begin
        terminal = BayesianSR.sampleterminal(tree, fullgrammar)
        operator = BayesianSR.sampleoperator(tree, fullgrammar)
        terminal = get(tree, terminal).ind
        operator = get(tree, operator).ind
        @test nchildren(fullgrammar, terminal) == 0
        @test in(nchildren(fullgrammar, operator), [1,2])
    end 
end 

@testset "Random node initialization" begin
    tree = BayesianSR.EqTree(fullgrammar).S
    test_tree(tree)
end 

function test_sample(sample::BayesianSR.Sample)
    @testset "Samples" begin
        @test length(sample.trees) == k
        for tree in sample.trees
            test_tree(tree.S)
        end 
        @test length(sample.β) == k+1
    end 
end 

@testset "Random sample initialization" begin
    @unpack σ²_prior = hyper
    Random.seed!(3)
    k = 3
    sample = BayesianSR.Sample(k, fullgrammar, σ²_prior)
    test_sample(sample)
    @test maximum(sample.β) == 0
    BayesianSR.optimβ!(sample, x, y, fullgrammar)
    @test length(sample.β) == k+1
    @test in(0, sample.β) == false
end 

function test_chain(chain::Chain)
    @test length(chain) == 1
    @test length(chain) == length(chain.samples)
    @test chain.samples[1] == chain.samples[end]
    test_sample(chain.samples[1])

    stat_keys = keys(chain.stats)
    @test length(stat_keys) == 2
    @test :lastj in stat_keys
    @test chain.stats[:lastj] <= chain.hyper.k
    @test :proposals in stat_keys
    @test chain.stats[:proposals]+1 >= length(chain)

    test_hyperparams(chain.hyper)
end 

@testset "Random Chain initialization" begin
    k = 3
    chain = Chain(x, y)
    test_chain(chain)
end 

@testset "Utils" begin 
    @testset "flatten()" begin
        node = RuleNode(1, [RuleNode(1, [RuleNode(1), RuleNode(1)])])
        flat = BayesianSR.flatten(node)
        @test length(flat) == 4
        @test 1 in flat 
        @test findall(x -> x == 1, flat) |> length == length(flat)
    end 
    @testset "sampleterminal()" begin
        for i in 1:20
            node = BayesianSR.EqTree(fullgrammar).S
            loc = BayesianSR.sampleterminal(node, fullgrammar)
            node = get(node, loc)
            @test node.ind in BayesianSR.terminal_indices(fullgrammar)
        end 
    end 
    @testset "sampleoperator()" begin
        for i in 1:20
            node = BayesianSR.EqTree(fullgrammar).S
            loc = BayesianSR.sampleoperator(node, fullgrammar)
            node = get(node, loc)
            @test node.ind in BayesianSR.operator_indices(fullgrammar)
        end 
    end 
    @testset "iscandidate()" begin
        node1 = RuleNode(1, [RuleNode(9), RuleNode(9)])
        @test BayesianSR.iscandidate(node1, node1, fullgrammar) == false
        node2 = RuleNode(1, [RuleNode(1, [RuleNode(9), RuleNode(9)]), RuleNode(9)])
        @test BayesianSR.iscandidate(node2, node2, fullgrammar)
        node3 = RuleNode(1, [RuleNode(1, [RuleNode(9), RuleNode(9)]),
                             RuleNode(1, [RuleNode(9), RuleNode(9)])])
        @test BayesianSR.iscandidate(node3, node3, fullgrammar)
        @test BayesianSR.iscandidate(RuleNode(9), node3, fullgrammar) == false
        @test BayesianSR.iscandidate(RuleNode(1, [RuleNode(9), RuleNode(9)]), node3, fullgrammar)
    end

    @testset "samplecandidate()" begin
        node1 = RuleNode(1, [RuleNode(9), RuleNode(9)])
        @test_throws ErrorException BayesianSR.samplecandidate(node1, fullgrammar)
        node2 = RuleNode(1, [RuleNode(1, [RuleNode(9), RuleNode(9)]), RuleNode(9)])
        for i in 1:10
            @test BayesianSR.samplecandidate(node2, fullgrammar).i in [0, 1]
        end 
        node3 = RuleNode(1, [RuleNode(1, [RuleNode(9), RuleNode(9)]),
                             RuleNode(1, [RuleNode(9), RuleNode(9)])])
        for i in 1:10
            @test BayesianSR.samplecandidate(node3, fullgrammar).i in [0, 1, 2]
        end 
    end 
end 

@testset "Tree movements" begin
    @testset "grow!()" begin
        node = BayesianSR.EqTree(fullgrammar).S
        old_length = length(BayesianSR.flatten(node))
        proposal = BayesianSR.grow!(node, fullgrammar)
        new_length = length(BayesianSR.flatten(proposal.tree))
        @test new_length >= old_length
        test_tree(proposal.tree)
        @test proposal.changed_node.ind in BayesianSR.terminal_indices(fullgrammar)
    end 

    @testset "prune!()" begin
        node = BayesianSR.EqTree(fullgrammar).S
        old_length = length(BayesianSR.flatten(node))
        proposal = BayesianSR.prune!(node, fullgrammar)
        new_length = length(BayesianSR.flatten(proposal.tree))
        @test new_length < old_length
        test_tree(proposal.tree)
        @test proposal.changed_node.ind in BayesianSR.operator_indices(fullgrammar)
    end 

     @testset "delete!()" begin
         function new_deleteable_node()
             node = BayesianSR.EqTree(fullgrammar).S
             try BayesianSR.samplecandidate(node, fullgrammar)
             catch e
                 node = new_node()
             end 
             return node
         end 
         node = new_deleteable_node()
         old_length = length(BayesianSR.flatten(node))
         proposal = BayesianSR.delete!(node, fullgrammar)
         new_length = length(BayesianSR.flatten(proposal.tree))
         @test new_length < old_length
         test_tree(proposal.tree)
         for i in 1:10
             node2 = RuleNode(1, [RuleNode(2, [RuleNode(8), RuleNode(8)]), RuleNode(9)])
             proposal2 = BayesianSR.delete!(node2, fullgrammar)
             if proposal2.dropped_node == RuleNode(8)
                 @test proposal2.p_child == 0.5
             else 
                 @test proposal2.p_child == 1
             end 
         end 
         node3 = RuleNode(1, [RuleNode(2, [RuleNode(9), RuleNode(9)]),
                              RuleNode(3, [RuleNode(9), RuleNode(9)])])
         proposal3 = BayesianSR.delete!(node3, fullgrammar)
         @test proposal3.p_child == 0.5
     end 

    @testset "insert_node!()" begin
        node = BayesianSR.EqTree(fullgrammar).S
        old_length = length(BayesianSR.flatten(node))
        proposal = BayesianSR.insert_node!(node, fullgrammar)
        new_length = length(BayesianSR.flatten(proposal.tree))
        @test new_length > old_length
        test_tree(proposal.tree)
    end 

    @testset "re_operator!()" begin
        node = BayesianSR.EqTree(fullgrammar).S
        old_node = deepcopy(node)
        old_length = length(BayesianSR.flatten(node))
        proposal = BayesianSR.re_operator!(node, fullgrammar)
        node = proposal.tree
        new_length = length(BayesianSR.flatten(node))
        @test node != old_node
        test_tree(node)
    end 

    @testset "re_feature!()" begin
        node = BayesianSR.EqTree(fullgrammar).S
        old_node = deepcopy(node)
        old_length = length(BayesianSR.flatten(node))
        node = BayesianSR.re_feature!(node, fullgrammar)
        new_length = length(BayesianSR.flatten(node))
        @test new_length == old_length
        @test node != old_node
        test_tree(node)
    end 
end 

# TODO: Tree sampling
# TODO: mcmc
