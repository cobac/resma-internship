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

const fullgrammar = append!(deepcopy(defaultgrammar), variablesgrammar)
