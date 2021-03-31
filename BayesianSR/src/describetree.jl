function flatten(node::RuleNode)
    out = []
    queue = [node]
    while !isempty(queue)
        push!(out, queue[1].ind)
        append!(queue, queue[1].children)
        deleteat!(queue, 1)
    end
    return out
end
