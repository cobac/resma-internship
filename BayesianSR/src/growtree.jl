"""
    growtree(grammar::Grammar, d::Int)

Samples a branch from the prior distribution.
`d` is the depth of the branch in a tree.
The root node is at `d = 1`.
"""
function growtree(grammar::Grammar, d::Int)
    node_types = nodetypes(grammar)
    node = RuleNode(0)
    
    # Hyper: α, β = 2, 1
    # Prior: Uniform for operators and features
    p₁ = 2/(1+d)
    if p₁ > rand()
        # Node = operator
        node.ind = sample(findall(x -> x==1 || x==2, node_types))
        for child in 1:node_types[node.ind]
            push!(node.children, growtree(grammar, d+1))
        end 
    else
        # Node = terminal
        node.ind = sample(findall(x -> x==0, node_types))
    end 

    return node
end 

"""
    growtree(grammar::Grammar) = growtree(grammar, 1)   

Samples a new tree.
"""
growtree(grammar::Grammar) = growtree(grammar, 1)
