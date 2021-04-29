function step!(chain::Chain)
    j = chain.stats[:lastj] + 1
    j == no_trees(chain) + 1 ? j = 1 : nothing
    chain.stats[:lastj] = j
    chain.stats[:proposals] += 1

    @unpack ν, λ = chain.hyper

    # Initialize new sample
    old_sample = deepcopy(chain.samples[end])
    proposal = deepcopy(chain.samples[end])

    # Propose a new tree
    proposal_tree = proposetree(proposal.trees[j], chain.grammar)

    # Propose a new set of parameters
    # TODO: When lt() implemented 

    # Update new sample
    proposal.trees[j] = proposal_tree.eqtree
    try 
        optimβ!(proposal, chain.x, chain.y, chain.grammar)
    catch e 
        println("Got an eval error!")
        return NaN
    end 
    σ²_prior = InverseGamma(ν / 2, ν * λ / 2)
    proposal.σ² = rand(σ²_prior)

    # Calculate R
    tree_x_proposal = evalsample(proposal, chain.x, chain.grammar)
    X_proposal = [ones(size(tree_x_proposal)[1]) tree_x_proposal]
    ε_proposal = chain.y - X_proposal * proposal.β
    L_proposal = sum(log.(pdf(Normal(0, √proposal.σ²), ε_proposal)))
    numerator = L_proposal +
        log(pdf(σ²_prior, proposal.σ²)) +
        sum([tree_p(eqtree.S, chain.grammar) for eqtree in proposal.trees]) +
        proposal_tree.p_mov

    tree_x_old_sample = evalsample(old_sample, chain.x, chain.grammar)
    X_old_sample = [ones(size(tree_x_old_sample)[1]) tree_x_old_sample]
    ε_old_sample = chain.y - X_old_sample * old_sample.β
    L_old_sample = sum(log.(pdf(Normal(0, √old_sample.σ²), ε_old_sample)))
    denominator = L_old_sample +
        log(pdf(σ²_prior, old_sample.σ²)) +
        sum([tree_p(eqtree.S, chain.grammar) for eqtree in old_sample.trees]) +
        proposal_tree.p_mov_inv
    
    R = exp(numerator - denominator)
    
    # Update chain
    α = min(1, R)
    if (rand() < α)
        println("Updating! Movement = ", proposal_tree.movement)
        push!(chain.samples, proposal)
    end 

    return R
end 


