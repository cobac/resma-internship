
"""
    LinearCoef(a::Float64, b::Float64)

Coefficients of a linear operator `a + b*x`.
"""
struct LinearCoef
    a::Float64
    b::Float64
end 

function LinearCoef(hyper::Hyperparams) 
    @unpack σ²_a_prior , σ²_b_prior = hyper
    σ²_a = rand(σ²_a_prior)
    σ²_b = rand(σ²_b_prior)
    a = rand(Normal(1, σ²_a))
    b = rand(Normal(0, σ²_b))
    return LinearCoef(a, b)
end 

function LinearCoef(σ²_a::AbstractFloat, σ²_b::AbstractFloat)
    a = rand(Normal(1, σ²_a))
    b = rand(Normal(0, σ²_b))
    return LinearCoef(a, b)
end 
