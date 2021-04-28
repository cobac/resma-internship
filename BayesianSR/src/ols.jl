function optimβ!(sample, x, y, grammar)
    #= Minimizes sample.β by OLS =#
    n = size(x)[1]
    xs = Matrix{Float64}(undef, (n, length(sample.trees)))
    for k in 1:length(sample.trees)
        xs[:, k] = evaltree(sample.trees[k], x, grammar)
    end 
    sample.β = ols(y, xs)
    return nothing
end 

function ols(y, x)
    X = [ones(size(x)[1]) x]
    # QR decomposition
    β = X \ y
    return β
end 
