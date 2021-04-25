struct TreeProposal
    eqtree::EqTree
    movement::Symbol
    p_mov::Float64
    # movement_inv::Symbol
    p_mov_inv::Float64
end 

function proposetree(tree::EqTree, grammar)
    tree = deepcopy(tree.S)
    nₒ = n_operators(tree, grammar)
    nₜ = n_terminals(tree, grammar)
    n_cand = n_candidates(tree, grammar)
    node_types = nodetypes(grammar)
    operator_is = findall(x -> x==1 || x==2, node_types)
    terminal_is = findall(x -> x==0, node_types)
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

    # Hyper: α, β = 2, 1
    # Prior: Uniform for operators and features
    if mov == :stay
        # Should never return until lt() are introduced
        p = p_inv = log(p_0)
    elseif mov == :grow
        changed_tree = grow!(tree, grammar)
        tree = changed_tree.tree
        p = log(p_g) + log(1/nₜ) + tree_p(changed_tree.changed_node, changed_tree.d, grammar)
        new_nₒ = n_operators(tree, grammar)
        new_p_0 = 0
        new_p_g = (1 - new_p_0)/3 * min(1, 8/(new_nₒ+2))
        new_p_p = (1-new_p_0)/3-new_p_g
        p_inv = log(new_p_p) +
            log(1/new_nₒ) +
            log(1/length(terminal_is))
    elseif mov == :prune
        changed_tree = prune!(tree, grammar)
        tree = changed_tree.tree
        p = log(p_p) + log(1/nₒ) + log(1/length(terminal_is))
        new_nₒ = n_operators(tree, grammar)
        new_p_0 = 0
        new_p_g = (1 - new_p_0)/3 * min(1, 8/(new_nₒ+2))
        p_inv = log(new_p_g) +
            log(1/n_terminals(tree, grammar)) +
            tree_p(changed_tree.changed_node, changed_tree.d, grammar)
    elseif mov == :delete
        deleted_tree = delete!(tree, grammar)
        tree = deleted_tree.tree
        p = log(p_d) + log(1/n_cand) + log(deleted_tree.p_child)
        new_n_cand = n_candidates(tree, grammar)
        new_p_0 = 0
        new_p_d = (1-p_0)/3 * new_n_cand/(new_n_cand+3)
        new_p_i = (1-new_p_0)/3 - new_p_d
        p_inv = log(new_p_i) +
            log(1/length(flatten(tree))) +
            log(1/length(operator_is)) +
            log(deleted_tree.p_child)
        if !isnothing(deleted_tree.changed_node)
            p_inv += tree_p(deleted_tree.changed_node, deleted_tree.d, grammar)
        end 
    elseif mov == :insert
        inserted_tree = insert_node!(tree, grammar)
        tree = inserted_tree.tree
        p = log(p_i) + log(1/length(flatten(tree))) +
            log(1/length(operator_is))
        if !isnothing(inserted_tree.new_branch)
            p += tree_p(inserted_tree.new_branch, inserted_tree.d, grammar)
        end 
        new_n_cand = n_candidates(tree, grammar)
        new_p_0 = 0
        new_p_d = (1-p_0)/3 * new_n_cand/(new_n_cand+3)
        p_inv = log(new_p_d) + log(1/new_n_cand)
        if !isnothing(inserted_tree.new_branch)
            p_inv += log(1/2)
        end 
    elseif mov == :re_operator
        tree = re_operator!(tree, grammar)
        p = p_inv = log(p_ro) + log(1/nₒ) + log(1/(length(operator_is) - 1))
    elseif mov == :re_feature
        tree = re_feature!(tree, grammar)
        p = p_inv = log(p_rf) + log(1/nₜ) + log(1/(length(terminal_is) - 1))
    end 

    return TreeProposal(EqTree(tree), mov, p, p_inv)
end 

function proposeparameters()
    nothing
end 
