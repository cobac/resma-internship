
function evaltree(tree, x, grammar)
    #= Evaluates a tree into a vector for every variable x_j =#
    n = size(x)[1]
    out = Vector{Float64}(undef, n)
    for i in 1:n
        table = tableforeval(x, i, grammar) 
        eq = ExprRules.get_executable(tree.S, grammar)
        out[i] = Core.eval(table, eq)
    end 
    return out
end 

function tableforeval(x, i, grammar)
    #= Generates a symbol table ready for evaluation for x_i =#
    symboltable = ExprRules.SymbolTable(grammar)
    k = size(x)[2]
    for m in 1:k
        symboltable[Symbol("x", m)] = x[i, m]
    end 
    return symboltable
end 
