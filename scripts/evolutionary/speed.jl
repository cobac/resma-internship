# Without the @everywhere hack all workers try to compile SymbolicRegression.jl at the same time and ... something bad happens
# https://stackoverflow.com/questions/60934852/why-does-multiprocessing-julia-break-my-module-imports
using Distributed, Pkg
@everywhere using Distributed, Pkg
Pkg.activate(".")
@everywhere Pkg.activate(".")

using SymbolicRegression,
    Random,
    Distributions,
    Symbolics,
    Symbolics,
    JLD2,
    Optim,
    FileIO
@everywhere using SymbolicRegression

Random.seed!(3)
n = 30 # no. observations
m = 2 # no. features
k = 2 # no. trees per sample
x = rand(Uniform(-3 , 3), (m, n))
   
f₁(x) = 2.5*x[1]^4 - 1.3*x[1]^3 + 0.5*x[2]^2 - 1.7*x[2]
y = f₁.(eachcol(x))

sq(x) = x^2
cb(x) = x^3

opt = Options(binary_operators = (+, -, *, /),
              unary_operators = (sin, cos, exp, sq, cb),
              progress = true,
              recorder = false,
              npop = 1000)

time = @elapsed EquationSearch(x, y, niterations = 10, runtests = false, options = opt)
# 364s
