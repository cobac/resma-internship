"""
   optimβ!(sample::Sample, x, y, grammar::Grammar) 

Optimises the β coefficients of a `Sample` by Ordinary Least Squares.

See also: `ols`
"""
function optimβ!(sample::Sample, x, y, grammar::Grammar)
    n = size(x)[1]
    xs = Matrix{Float64}(undef, (n, length(sample.trees)))
    for k in 1:length(sample.trees)
        xs[:, k] = evaltree(sample.trees[k], x, grammar)
    end 
    sample.β = ols(y, xs)
    return nothing
end 

"""
    ols(y, x)

Ordinary Least Squares via QR decomposition.
"""
function ols(y, x)
    X = [ones(size(x)[1]) x]
    β = X \ y
    return β
end 
