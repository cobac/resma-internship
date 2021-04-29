
function evaltree(tree::EqTree, x, grammar::Grammar)
    #= Evaluates a tree into a vector for every variable x_j =#
    n = size(x)[1]
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        table = tableforeval(x, i, grammar) 
        eq = ExprRules.get_executable(tree.S, grammar)
        out[i] = Core.eval(table, eq)
    end 
    return out
end 

function evalsample(sample::Sample, x, grammar::Grammar)
    out = Matrix{Float64}(undef, size(x)[1], length(sample.trees))
    @inbounds for j in 1:length(sample.trees)
        out[:, j] = evaltree(sample.trees[j], x, grammar)
    end 
    return out
end 

function tableforeval(x, i, grammar::Grammar)
    #= Generates a symbol table ready for evaluation for x_i =#
    symboltable = ExprRules.SymbolTable(grammar)
    k = size(x)[2]
    @inbounds for m in 1:k
        symboltable[Symbol("x", m)] = x[i, m]
    end 
    return symboltable
end 
