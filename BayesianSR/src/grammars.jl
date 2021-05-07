"""
`Grammar` with default operators.
"""
const defaultgrammar = @grammar begin
    Real = Real + Real
    Real = Real - Real
    Real = Real * Real 
    Real = Real / Real
    Real = cos(Real) 
    Real = sin(Real) 
end

"""
    variablestogrammar(x)

Creates a `Grammar` with all the features in `x`.
"""
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

"""
    nodetypes(grammar::Grammar)

Returns a vector with the types of possible nodes from a `Grammar`.
- 1: unary operator
- 2: binary operator
- 0: terminal node
"""
function nodetypes(grammar::Grammar)
    types = [nchildren(grammar, i)
             for i in 1:length(grammar)]
    return types
end 

"""
    operator_indices(grammar::Grammar)

Returns a vector with the indices of all operators in a grammar.
"""
function operator_indices(grammar::Grammar)
    node_types = nodetypes(grammar)
    is = findall(x -> x==1 || x==2, node_types)
    return is
end 

"""
    terminal_indices(grammar::Grammar)

Returns a vector with the indices of all terminals in a grammar.
"""
function terminal_indices(grammar::Grammar)
    node_types = nodetypes(grammar)
    is = findall(x -> x==0, node_types)
    return is
end 

# This is the definition found in ExprRules.
# It started not being imported properly again, so I've just redefined it for now. 
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

