struct ChangedTree
    tree::RuleNode
    changed_node::RuleNode
    d::Int64
end 

function grow!(node::RuleNode, grammar::Grammar)
    loc = sampleterminal(node, grammar)
    old_node = get(node, loc)
    d = node_depth(node, old_node)
    new_node = growtree(grammar, d)
    insert!(node, loc, new_node)
    return ChangedTree(node, new_node, d)
end 

function prune!(node::RuleNode, grammar::Grammar)
    node_types = nodetypes(grammar)
    terminal_is = findall(x -> x==0, node_types)
    loc = sampleoperator(node, grammar)
    old_node = get(node, loc)
    d = node_depth(node, old_node)
    insert!(node, loc, RuleNode(sample(terminal_is)))
    return ChangedTree(node, old_node, d)
end 

struct DeletedTree
    tree::RuleNode
    changed_node::Union{Nothing, RuleNode}
    d::Int64 # Union{Nothing, Int64}
    p_child::Float64
end 

function delete!(node::RuleNode, grammar::Grammar)
    loc = samplecandidate(node, grammar)
    target = get(node, loc)
    d = node_depth(node, target)
    n_children = length(target.children)
    if n_children == 1
        p_child = 1
        changed_node = nothing
        if loc.i == 0 
            node = target.children[1]
        else 
            insert!(node, loc, target.children[1])
        end 
    elseif target != node # no unary and no root = binary no root
        p_child = 0.5
        i = sample(1:2)
        changed_node = target.children[i == 1 ? 2 : 1]
        insert!(node, loc, target.children[i])
    else # only binary root left
        node_types = nodetypes(grammar)
        operator_is = findall(x -> x==1 || x==2, node_types)
        ind_children = [child.ind for child in target.children]
        ind_operator = findall(x -> in(x, operator_is), ind_children)
        p_child = 1/length(ind_operator)
        i = sample(ind_operator)
        if length(ind_operator) == 2
            changed_node = target.children[i == 1 ? 2 : 1]
        else
            changed_node = nothing
        end 
        node = target.children[i]
    end 
    return DeletedTree(node, changed_node, d, p_child)
end 

struct InsertedTree
    tree::RuleNode
    new_branch::Union{Nothing, RuleNode}
    d::Int64 # Union{Nothing, Int64}
end 

function insert_node!(node::RuleNode, grammar::Grammar)
    node_types = nodetypes(grammar)
    operator_is = findall(x -> x==1 || x==2, node_types)
    loc = sample(NodeLoc, node)
    old = get(node, loc)
    new = RuleNode(sample(operator_is))
    type = node_types[new.ind]
    d = node_depth(node, old)
    if type == 2
        new_branch = growtree(grammar, d)
        new.children = [old, new_branch]
    else # type == 1
        push!(new.children, old)
        new_branch = nothing
    end 
    if loc.i == 0
        node = new
    else 
        node = insert!(node, loc, new)
    end 
    return InsertedTree(node, new_branch, d)
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
