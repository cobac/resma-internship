
function grow!(node::RuleNode, grammar::Grammar)
    loc = sampleterminal(node, grammar)
    d = node_depth(node, get(node, loc))
    insert!(node, loc, growtree(grammar, d))
    return node
end 

function prune!(node::RuleNode, grammar::Grammar)
    node_types = nodetypes(grammar)
    terminal_is = findall(x -> x==0, node_types)
    loc = sampleoperator(node, grammar)
    insert!(node, loc, RuleNode(sample(terminal_is)))
    return node
end 

function delete!(node::RuleNode, grammar::Grammar)
    loc = samplecandidate(node, grammar)
    target = get(node, loc)
    n_children = length(target.children)
    if n_children == 1
        insert!(node, loc, target.children[1])
    elseif target != node # no unary and no root = binary no root
        insert!(node, loc, sample(target.children))
    else # only binary root left
        node_types = nodetypes(grammar)
        operator_is = findall(x -> x==1 || x==2, node_types)
        ind_children = [child.ind for child in target.children]
        ind_operator = findall(x -> in(x, operator_is), ind_children)
        node = target.children[sample(ind_operator)]
        # insert!(node, loc, target.children[sample(ind_operator)])
    end 
    return node 
end 

function insert_node!(node::RuleNode, grammar::Grammar)
    node_types = nodetypes(grammar)
    operator_is = findall(x -> x==1 || x==2, node_types)
    loc = sample(NodeLoc, node)
    old = get(node, loc)
    new = RuleNode(sample(operator_is))
    type = node_types[new.ind]
    println("old is ", old)
    println("new is ", new)
    println("Type is ", type)
    if type == 2
        d = node_depth(node, old)
        new.children = [old, growtree(grammar, d)]
    else # type == 1
        push!(new.children, old)
    end 
    insert!(node, loc, new)
    return node
end 

function re_operator!(node::RuleNode, grammar::Grammar)
    node_types = nodetypes(grammar)
    operator_is = findall(x -> x==1 || x==2, node_types)
    loc = sampleoperator(node, grammar)
    target = get(node, loc)
    old_ind = target.ind
    old_type = node_types[old_ind]
    # Remove old index from the pool of operators
    operator_rm = findfirst(isequal(target.ind), operator_is)
    deleteat!(operator_is, operator_rm)
    # Sample new operator
    new_ind = sample(operator_is)
    new_type = node_types[new_ind]
    new = RuleNode(new_ind)
    target.ind = new_ind
    if new_type == 2 && old_type == 1
        d = node_depth(node, target)
        push!(target.children, growtree(grammar, d))
    elseif new_type == 1 && old_type == 2
        deleteat!(target.children, 2)
    else # new_type == old_type
        nothing
    end 
    insert!(node, loc, target)
    return node
end 

function re_feature!(node::RuleNode, grammar::Grammar)
    node_types = nodetypes(grammar)
    terminal_is = findall(x -> x==0, node_types)
    loc = sampleterminal(node, grammar)
    target = get(node, loc)
    # Remove old index from the pool of terminals
    terminal_rm = findfirst(isequal(target.ind), terminal_is)
    deleteat!(terminal_is, terminal_rm)
    # Sample new terminal
    target.ind = sample(terminal_is)
    insert!(node, loc, target)
    return node
end 
