using BayesianSR,
    Random,
    Distributions,
    BenchmarkTools

Random.seed!(3)
n = 30 # no. observations
m = 2 # no. features
k = 2 # no. trees per sample
x = rand(Uniform(-3 , 3), (n, m))
   
f₁(x)=2.5*x[1]^4 - 1.3*x[1]^3 + 0.5*x[2]^2 - 1.7*x[2]

y = f₁.(eachrow(x))
 
functions = @grammar begin
    Real = Real + Real
    Real = Real - Real
    Real = Real * Real 
    Real = Real / Real 
    Real = sin(Real) 
    Real = cos(Real) 
    Real = exp(Real)
    Real = Real^2
    Real = Real^3
end

# Each equation is a linear combination of k=2 symbolic trees
hyper = Hyperparams(k=2)

@benchmark mcmc!(chain, 1000) setup = (chain = Chain(x, y, operators = functions, hyper = hyper)) seconds = 20
# 495 ms / 1000 = 495 μs/sample
# Per run: 336 *  100,000 = 49.5 s
# Per simulation: 50 * 6 = 4.1h

