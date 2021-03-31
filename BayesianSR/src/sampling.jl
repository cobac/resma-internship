struct TreeProposal
    # k::Int ?? probably doesn't make sense here
    sample::Sample
    movements::Symbol[]
end 

function generateproposal()
    
end 

function proposetree(tree::RuleNode, grammar)
    nₒ = n_operators(tree, grammar)
    # nₜ= n_terminals(tree, grammar)
    n_cand = n_candidates(tree, grammar)

    # TODO: modify after implementing =lt()= operators
    # P_0 = P(tree stays the same)
    p_0 = 0
    if p_0 > rand()
        return tree
    end 

    tree = deepcopy(tree)

    # P_g = P(tree grows)
    p_g = (1 - p_0)/3 * min(1, 8/(nₒ+2))
    if p_g > rand()
        
    end 

    # P_p = P(pruning the tree)
    p_p = (1-p_0)/3-p_g
    if p_p > rand()

    end 

    # P_d = P(deleting a candidate node)
    p_d = (1-p_0)/3 * n_cand/(n_cand+3)
    if p_d > rand()

    end 

    # P_i = P(insert a new node)
    p_i  = (1-p_0)/3 - p_d
    if p_d > rand()

    end 

    # P_ro = P(reassign operator)
    # P_rf = P(reassign feature)
    p_ro = p_rf = (1-p_0)/6
    if p_ro > rand()

    end 

    if p_rf > rand()

    end 

    return (eqtree = tree, movements = movements)

end 




