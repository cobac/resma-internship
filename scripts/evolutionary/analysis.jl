using Random, Distributions, StatsBase, JLD2

Random.seed!(3)
n = 30 # no. observations
m = 2 # no. features
x = rand(Uniform(-3 , 3), (n, m))

f₁(x) = 2.5*x[1]^4 - 1.3*x[1]^3 + 0.5*x[2]^2 - 1.7*x[2]
f₂(x) = 8*x[1]^2 + 8*x[2]^3 - 15
f₃(x) = 0.2*x[1]^3 + 0.5*x[2]^3 - 1.2*x[2] - 0.5*x[1]
f₄(x) = 1.5*exp(x[1]) + 5*cos(x[2])
f₅(x) = 6 * sin(x[1]) * cos(x[2])
f₆(x) = 1.35*x[1]*x[2] * 5.5*sin((x[1]-1)*(x[2]-1))

y = Matrix{Float64}(undef, n, 6)
y[:, 1] = f₁.(eachrow(x))
y[:, 2] = f₂.(eachrow(x))
y[:, 3] = f₃.(eachrow(x))
y[:, 4] = f₄.(eachrow(x))
y[:, 5] = f₅.(eachrow(x))
y[:, 6] = f₆.(eachrow(x))

# Change of seed to keep the testing sets consistent across models
Random.seed!(1)
x_test₁ = rand(Uniform(-3 , 3), (n, m))
x_test₂ = rand(Uniform(-6 , 6), (n, m))
x_test₃ = rand(Uniform(3 , 6),  (n, m))

sq(x) = x^2
cb(x) = x^3


# Get total number of simulations and equations
# Only works for 1 simulation atm
filenames = readdir("./hofs")
rx_files = r"hofs-(\d+)-(\d+)$"
no_eqs = 0
for file in filenames
    cap = match(rx_files, file)
    if !isnothing(cap)
        global no_eqs = max(no_eqs, parse(Int, cap.captures[2]))
    end 
end 

# 11 = max no. of equations in a hall of fame
rmse_train = zeros(Float64, 11, no_eqs)
rmse_test₁ = zeros(Float64, 11, no_eqs)
rmse_test₂ = zeros(Float64, 11, no_eqs)
rmse_test₃ = zeros(Float64, 11, no_eqs)

rx_mse_expr = r"\d+\|(.*)\|(.*)$"
eval_f(fun, matrix) = (x -> fun(x[1], x[2])).(collect(eachrow(matrix)))

for eq in 1:no_eqs
    lines = readlines(string("./hofs/hofs-1-", eq))
    for line in 2:length(lines)
        cap = match(rx_mse_expr, lines[line])
        # Recover RMSE from the training set
        rmse_train[line-1, eq] = sqrt(parse(Float64, cap.captures[1]))
        # Recover equation to get the RMSE for the test sets
        body = Meta.parse(cap.captures[2])
        head = :(f(x1, x2))
        fun = Expr(:(=), head, body) |> eval
        ŷ₁ = eval_f(fun, x_test₁)
        ŷ₂ = eval_f(fun, x_test₂)
        ŷ₃ = eval_f(fun, x_test₃)
        rmse_test₁[line-1, eq] = rmsd(y[:, eq], ŷ₁)
        rmse_test₂[line-1, eq] = rmsd(y[:, eq], ŷ₂)
        rmse_test₃[line-1, eq] = rmsd(y[:, eq], ŷ₃)
    end        
end 

jldsave("output.jld2"; rmse_train, rmse_test₁, rmse_test₂, rmse_test₃)
