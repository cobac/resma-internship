using SymbolicRegression,
    Random,
    Distributions,
    Symbolics,
    Symbolics,
    JLD2,
    Optim,
    FileIO

Random.seed!(3)
n = 30 # no. observations
m = 2 # no. features
k = 2 # no. trees per sample
x = rand(Uniform(-3 , 3), (m, n))
   
f₁(x) = 2.5*x[1]^4 - 1.3*x[1]^3 + 0.5*x[2]^2 - 1.7*x[2]
f₂(x) = 8*x[1]^2 + 8*x[2]^3 - 15
f₃(x) = 0.2*x[1]^3 + 0.5*x[2]^3 - 1.2*x[2] - 0.5*x[1]
f₄(x) = 1.5*exp(x[1]) + 5*cos(x[2])
f₅(x) = 6 * sin(x[1]) * cos(x[2])
f₆(x) = 1.35*x[1]*x[2] * 5.5*sin((x[1]-1)*(x[2]-1))

# AbstractFloat generates an error on the loss functions
y = Matrix{Float64}(undef, n, 6)
y[:, 1] = f₁.(eachcol(x))
y[:, 2] = f₂.(eachcol(x))
y[:, 3] = f₃.(eachcol(x))
y[:, 4] = f₄.(eachcol(x))
y[:, 5] = f₅.(eachcol(x))
y[:, 6] = f₆.(eachcol(x))

sq(x) = x^2
cb(x) = x^3

for sim=1:10, expr=1:6
    println("=================================")
    println("Sim. no. ", sim, " . Expression no. ", expr)
    opt = Options(binary_operators = (+, -, *, /),
                  unary_operators = (sin, cos, exp, sq, cb),
                  progress = false,
                  recorder = true,
                  npopulations = 5,
                  hofFile = string("./hofs/hofs-", sim, "-", expr),
                  npop = 1000,
                  recorder_file = string("./json/recorder-", sim, "-", expr, ".json"))

    hof = EquationSearch(x, y[:, expr],
                         niterations = 2,
                         numprocs = 5,
                         runtests = false,
                         options = opt)
end 

# No. samples = niterations * ncyclesperiteration * npopulations * npop
#                             (300                 5           1000)
#                niterations *  1500000
