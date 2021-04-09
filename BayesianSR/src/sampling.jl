struct TreeProposal
    sample::Sample
    movement::Symbol
end 

function proposetree(tree::RuleNode, grammar)
    tree = deepcopy(tree)
    nₒ = n_operators(tree, grammar)
    n_cand = n_candidates(tree, grammar)
    # TODO: modify after implementing =lt()= operators
    # P_0 = P(tree stays the same)
    p_0 = 0
    # P_g = P(tree grows)
    p_g = (1 - p_0)/3 * min(1, 8/(nₒ+2))
    # P_p = P(pruning the tree)
    p_p = (1-p_0)/3-p_g
    # P_d = P(deleting a candidate node)
    p_d = (1-p_0)/3 * n_cand/(n_cand+3)
    # P_i = P(insert a new node)
    p_i  = (1-p_0)/3 - p_d
    # P_ro = P(reassign operator)
    # P_rf = P(reassign feature)
    p_ro = p_rf = (1-p_0)/6
    movements = [:stay , :grow, :prune, :delete, :insert, :re_operator, :re_feature]
    weights =   [p_0   ,   p_g,    p_p,     p_d,     p_i,         p_ro,        p_rf]
    mov = sample(movements, Weights(weights))

    if mov == :stay
        nothing
    elseif mov == :grow
        tree = grow!(tree)
    elseif mov == :prune
        tree = prune!(tree)
    elseif mov == :delete
        tree = delete!(tree)
    elseif mov == :insert
        tree = insert_node!(tree)
    elseif mov == :re_operator
        tree = re_operator!(tree)
    elseif mov == :re_feature
        tree = re_feature!(tree)
    end 

    return (tree = tree, movement = mov)
end 

function proposeparameters()
    nothing
end 



