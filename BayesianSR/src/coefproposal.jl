"""
    SetLinearCoef

Object with the set of LinearCoef of a RuleNode.
"""
struct SetLinearCoef
    a::Vector{AbstractFloat}
    b::Vector{AbstractFloat}
end 

"""
    proposeparameters!(tree::RuleNode)
Updates a tree with a new set of LinearCoef.
Returns an object of type SetLinearCoef with the new coefficients.

See also: SetLinearCoef
"""
function propose_LinearCoef!(tree::RuleNode, σ²_a::AbstractFloat, σ²_b::AbstractFloat)
    new_a = AbstractFloat[]
    new_b = AbstractFloat[]
    queue = [tree]
    while !isempty(queue)
        current = queue[1]
        if current.ind == 1 # LinearCoef
            current._val = LinearCoef(σ²_a, σ²_b)
            push!(new_a, current._val.a)
            push!(new_b, current._val.b)
        end 
        append!(queue, queue[1].children)
        deleteat!(queue, 1)
    end
    return SetLinearCoef(new_a, new_b)
end 

function recover_LinearCoef(tree::RuleNode)
    a = AbstractFloat[]
    b = AbstractFloat[]
    queue = [tree]
    while !isempty(queue)
        current = queue[1]
        if current.ind == 1 # LinearCoef
            push!(a, current._val.a)
            push!(b, current._val.b)
        end 
        append!(queue, queue[1].children)
        deleteat!(queue, 1)
    end
    return SetLinearCoef(a, b)
end 
