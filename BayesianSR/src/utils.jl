function flatten(node::RuleNode)
    out = Int[]
    queue = [node]
    while !isempty(queue)
        push!(out, queue[1].ind)
        append!(queue, queue[1].children)
        deleteat!(queue, 1)
    end
    return out
end

function sampleterminal(node::RuleNode, grammar::Grammar)
    node_types = nodetypes(grammar)
    operator_is = findall(x -> x==1 || x==2, node_types)
    out = sample(NodeLoc, node)
    # if node is operator, resample
    target = get(node, out)
    while in(target.ind, operator_is)
        out = sample(NodeLoc, node)
        target = get(node, out)
    end 
    return out
end 

function sampleoperator(node::RuleNode, grammar::Grammar)
    node_types = nodetypes(grammar)
    terminal_is = findall(x -> x==0, node_types)
    out = sample(NodeLoc, node)
    # if node is terminal, resample
    target = get(node, out)
    while in(target.ind, terminal_is)
        out = sample(NodeLoc, node)
        target = get(node, out)
    end 
    return out
end 
