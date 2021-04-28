const defaultgrammar = ExprRules.@grammar begin
    Real = Real + Real
    Real = Real - Real
    Real = Real * Real 
    Real = Real / Real
    Real = cos(Real) 
    Real = sin(Real) 
end

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
             for i in 1:length(grammar)]
    return types
end 

function Base.append!(grammar1::Grammar, grammar2::Grammar)
    N = length(grammar1.rules)
    append!(grammar1.rules, grammar2.rules)
    append!(grammar1.types, grammar2.types)
    append!(grammar1.isterminal, grammar2.isterminal)
    append!(grammar1.iseval, grammar2.iseval)
    append!(grammar1.childtypes, copy.(grammar2.childtypes))
    for (s,v) in grammar2.bytype
        grammar1.bytype[s] = append!(get(grammar1.bytype, s, Int[]), N .+ v)
    end
    grammar1
end

# TODO: Define this functions: operator_is, terminal_is
