
function operatortypes(grammar=defaultgrammar)
    #= Classifies operators into unary-binary-terminal =#
    types = [ExprRules.nchildren(grammar, i)
             for i in 1:length(grammar.rules)]
    return types
end 

function growtree!(node::RuleNode, grammar::Grammar, d::Int)
    operator_types = operatortypes(grammar)
    
    # Hyper: α, β = 2, 1
    # Prior: Uniform for operators and parameters
    p₁ = 2/(1+d)
    if p₁ > rand()
        # Node = operator
        node.ind = sample(findall(x -> x==1 || x==2, operator_types))
        for child in 1:operator_types[node.ind]
            push!(node.children, growtree!(grammar, d+1))
        end 
    else
        # Node = terminal
        node.ind = sample(findall(x -> x==0, operator_types))
    end 

    return node
end 

growtree!(node::RuleNode, grammar::Grammar) = growtree!(node, grammar, 1)
growtree!(grammar::Grammar) = growtree!(RuleNode(0), grammar, 1)
growtree!(grammar::Grammar, d::Int) = growtree!(RuleNode(0), grammar, d)
