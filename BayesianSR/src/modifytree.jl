function optimβ!(model, y, x, grammar)
    #= Minimizes model.β by OLS =#
    n = size(x)[1]
    xs = Matrix{Float64}(undef, (n, length(model.trees)))
    for k in 1:length(model.trees)
        xs[:, k] = evaltree(model.trees[k], x, grammar)
    end 
    model.β = ols(y, xs)
    return nothing
end 

function ols(y, x)
    X = [ones(size(x)[1]) x]
    β = inv(X' * X) * X' * y
    return β
end 

#node_index = sample(NodeLoc, tree)
#old_node = get(tree, node_index)
#new_node = rand(RuleNode, grammar, :Real, 3)
#insert!(tree, node_index, new_node)
#new_equation = get_executable(tree, grammar)
