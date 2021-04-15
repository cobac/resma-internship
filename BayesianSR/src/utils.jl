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

Base.length(grammar::Grammar) = length(grammar.rules)

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

function samplecandidate(node::RuleNode, grammar::Grammar)
    out = sample(NodeLoc, node)
    # if node is not candidate,  resample
    target = get(node, out)
    while !iscandidate(target, node, grammar)
        out = sample(NodeLoc, node)
        target = get(node, out)
    end 
    return out
end 

function iscandidate(target::RuleNode, root::RuleNode, grammar::Grammar)
    node_types = nodetypes(grammar)
    terminal_is = findall(x -> x==0, node_types)
    operator_is = findall(x -> x==1 || x==2, node_types)
    inds = [child.ind for child in target.children]
    if in(target.ind, terminal_is)
        # terminal node == false
        return false
    elseif target == root && count(i -> in(i, operator_is), inds) == 0
        # root node with all terminal children == false
        return false
    else
        return true
    end 
end 

Base.length(chain::Chain) = length(chain.samples)
no_trees(chain::Chain) = length(chain.samples[1].trees)

function n_operators(node::RuleNode, grammar::Grammar)
    node_types = nodetypes(grammar)
    operator_is = findall(x -> x==1 || x==2, node_types)
    nodes = flatten(node)
    out = count(i -> in(i, operator_is), nodes)
    return out
end

function n_terminals(node::RuleNode, grammar::Grammar)
    node_types = nodetypes(grammar)
    terminal_is = findall(x -> x==0, node_types)
    nodes = flatten(node)
    out = count(i -> in(i, terminal_is), nodes)
    return out
end

function n_candidates(node::RuleNode, grammar::Grammar)
    n = n_operators(node, grammar)
    node_types = nodetypes(grammar)
    operator_is = findall(x -> x==1 || x==2, node_types)
    inds = [child.ind for child in node.children]
    if count(i -> in(i, operator_is), inds) == 0
        n -= 1
    end 
    return n
end 
