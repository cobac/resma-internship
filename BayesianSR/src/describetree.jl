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
