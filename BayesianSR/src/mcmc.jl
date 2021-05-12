"""
    step(chain::Chain)

Generates a new Sample of `Chain`.
"""
function step(chain::Chain, i::Int, j::Int ; verbose::Bool = false)
    @unpack σ²_prior, σ²_a_prior, σ²_b_prior = chain.hyper

    # Initialize new sample
    old_sample = deepcopy(chain.samples[i])
    proposal = deepcopy(chain.samples[i])

    # Gather old LinearCoef parameters
    θ_old = recover_LinearCoef(proposal.trees[j])

    # Propose a new tree
    proposal_tree = proposetree(proposal.trees[j], chain.grammar, chain.hyper)

    # Propose a new set of LiearCoef parameters
    proposal.σ²[:σ²_a] = σ²_a = rand(σ²_a_prior)
    proposal.σ²[:σ²_b] = σ²_b = rand(σ²_b_prior)
    θ_new = propose_LinearCoef!(proposal.trees[j], σ²_a, σ²_b)

    # Update the proposal
    proposal.trees[j] = proposal_tree.tree
    try 
        optimβ!(proposal, chain.x, chain.y, chain.grammar)
    catch e 
        verbose && println("Got an eval error!")
        return NaN
    end 
    proposal.σ²[:σ²] = rand(σ²_prior)

    # TODO: Fix R calcualtion
    # # Calculate R
    # tree_x_proposal = evalsample(proposal, chain.x, chain.grammar)
    # X_proposal = [ones(size(tree_x_proposal)[1]) tree_x_proposal]
    # ε_proposal = chain.y - X_proposal * proposal.β
    # L_proposal = sum(logpdf.(Normal(0, √proposal.σ²), ε_proposal))
    # numerator = L_proposal + # Likelihood
    #     logpdf(σ²_prior, proposal.σ²) + # σ² prior
    #     # Trees prior
    #     sum([tree_p(tree, chain.grammar) for tree in proposal.trees]) +
    #     # Probability of tree jump
    #     proposal_tree.p_mov

    # tree_x_old_sample = evalsample(old_sample, chain.x, chain.grammar)
    # X_old_sample = [ones(size(tree_x_old_sample)[1]) tree_x_old_sample]
    # ε_old_sample = chain.y - X_old_sample * old_sample.β
    # L_old_sample = sum(logpdf.(Normal(0, √old_sample.σ²), ε_old_sample))
    # denominator = L_old_sample + # Likelihood
    #     logpdf(σ²_prior, old_sample.σ²) + # σ² prior 
    #     # Trees prior
    #     sum([tree_p(tree, chain.grammar) for tree in old_sample.trees]) +
    #     # Probability of tree jump
    #     proposal_tree.p_mov_inv
    # 
    # R = exp(numerator - denominator)
    R = rand()
    
    # Update chain
    α = min(1, R)
    if (rand() < α)
        verbose && println("Updating! Movement = ", proposal_tree.movement)
        return proposal
    else 
        verbose && println("Not updating! Movement = ", proposal_tree.movement)
        return old_sample
    end 
end 

function mcmc!(chain::Chain, n_steps::Int = 100; verbose::Bool = false)
    i₀ = length(chain)
    resize!(chain.samples, i₀ + n_steps)
    for i in (i₀ + 1):(i₀ + n_steps)
        j = chain.stats[:lastj] + 1
        j == no_trees(chain) + 1 ? j = 1 : nothing
        chain.stats[:lastj] = j
        chain.samples[i] = BayesianSR.step(chain, i-1, j, verbose = verbose)
    end 
end 
