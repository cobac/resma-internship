struct IndAndCount
    ind::Int64
    cnt::Int64
end 
"""
    flatten_with_depth(node::RuleNode, d::Int64)

Flattens a `RuleNode` as a vector of (indices, depth) starting at depth `d`.
"""
function flatten_with_depth(node::RuleNode, d::Int64)
    d < 1 && error("Initial d must be greater than 1.")
    out = IndAndCount[]
    queue = [RuleNodeAndCount(node, d)]
    while !isempty(queue)
        push!(out, IndAndCount(queue[1].node.ind, queue[1].cnt))
        append!(queue, RuleNodeAndCount.(queue[1].node.children, queue[1].cnt + 1))
        deleteat!(queue, 1)
    end
    return out
end

"""
    flatten_with_depth(node::RuleNode) = flatten_with_depth(node, 1)

Flattens a `RuleNode` as a vector of (indices, depth) starting at depth `d = 1`.
"""
flatten_with_depth(node::RuleNode) = flatten_with_depth(node, 1)

"""
    tree_p(node::RuleNode, d::Int64, grammar::Grammar)

Calculates the natural logarithm of the prior probability of a `RuleNode` starting at depth `d`.
"""
function tree_p(node::RuleNode, d::Int64, grammar::Grammar)
    nodes = flatten_with_depth(node, d)
    operator_is = operator_indices(grammar)
    terminal_is = terminal_indices(grammar)
    p = 0
    for node in nodes
        # Hyper: α, β = 2, 1
        # Prior: Uniform for operators and features
        if node.ind in operator_is
            p += log(2/(1 + node.cnt)) # P of inserting an operator
            + log(1/length(operator_is)) # P of selecting this operator
        else 
            p += log(1 - (2/(1 + node.cnt))) # P of inserting a terminal
            + log(1/length(terminal_is)) # P of selecting this terminal
        end 
    end 
    return p
end

"""
    tree_p(node::RuleNode, d::Int64, grammar::Grammar)

Calculates the natural logarithm of the prior probability of a `RuleNode` starting at depth `d = 1`.
"""
tree_p(node::RuleNode, grammar::Grammar) = tree_p(node, 1, grammar)
