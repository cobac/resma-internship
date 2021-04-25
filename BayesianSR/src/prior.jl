struct IndAndCount
    ind::Int64
    cnt::Int64
end 

function flatten_with_depth(node::RuleNode, d::Int64)
    d <= 0 && error("Initial d must be greater than 0.")
    out = IndAndCount[]
    queue = [RuleNodeAndCount(node, d)]
    while !isempty(queue)
        push!(out, IndAndCount(queue[1].node.ind, queue[1].cnt))
        append!(queue, RuleNodeAndCount.(queue[1].node.children, queue[1].cnt + 1))
        deleteat!(queue, 1)
    end
    return out
end

flatten_with_depth(node::RuleNode) = flatten_with_depth(node, 1)

function tree_p(node::RuleNode, d::Int64, grammar::Grammar)
    nodes = flatten_with_depth(node, d)
    node_types = nodetypes(grammar)
    operator_is = findall(x -> x==1 || x==2, node_types)
    terminal_is = findall(x -> x==0, node_types)
    p = 0
    for node in nodes
        # Hyper: α, β = 2, 1
        # Prior: Uniform for operators and features
        if node.ind in operator_is
            p += log(2/(1 + node.cnt)) + log(1/length(operator_is))
        else 
            p += log(1 - (2/(1 + node.cnt))) + log(1/length(terminal_is))
        end 
    end 
    return p
end

tree_p(node::RuleNode, grammar::Grammar) = tree_p(node, 1, grammar)
