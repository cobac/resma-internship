const defaultgrammar = ExprRules.@grammar begin
    Real = Real + Real
    Real = Real - Real
    Real = Real * Real 
    Real = Real / Real
    Real = cos(Real) 
    Real = sin(Real) 
end

# TODO: Ask for custom grammar

function variablestogrammar(x)
    k = size(x)[2]
    rules = [Symbol("x", i) for i in 1:k]
    types = [Symbol("Real") for i in 1:k]
    isterminal = ones(Int64, k)
    iseval = zeros(Int64, k)
    childtypes = [Symbol[] for i in 1:k]
    bytype = Dict(:Real => [i for i in 1:k])
    
    grammar = Grammar(rules, types, isterminal, iseval, bytype, childtypes)
    return grammar
end 

function nodetypes(grammar::Grammar)
    types = [ExprRules.nchildren(grammar, i)
             for i in 1:length(grammar.rules)]
    return types
end 
