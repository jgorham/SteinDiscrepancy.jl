# Computes classical Stein discrepancy based on the g and gradg coefficients
# for univariate target distributions:
#
#     max_{g} sum_i gobjcoeff(i) * g(x(i)) + gradgobjcoeff(i) * g'(x(i)))
#     subject to, for all z, y
#     |g(z)| <= c_{1} I[alpha < z < beta], |g'(z)| <= c_{2},
#     |g(z) - g(y)| <= c_{2} |z-y|,
#     |g'(z) - g'(y)| <= c_3 |z-y|,
#     |g(z) - g(y) - g'(y)(z - y)| <= (c_3/2) |z-y|^2,
#     |g(z) - g(y) - g'(z)(z - y)| <= (c_3/2) |z-y|^2
#    where (alpha,beta) is the support of target distribution.
#
# Despite this being an infinite dimensional feasible set, there
# is a set of finite dimensional constraints that can exactly
# recover the optimal solution g, g' on the points x_i.
#
# Args:
#  sample - SteinDiscrete object representing a sample
#  gobjcoefficients - a vector of coefficients for the objective for g
#  gradgobjcoefficients - a vector of coefficients for the objective for gprime
#  solver - optimization program solver supported by JuMP that can solve
#   a quadratically constrained quadratic program (QCQP)
#  operator - the operator being used for the objective
function affine_classical_discrepancy(sample::SteinDiscrete,
                                      gobjcoefficients::Array{Float64,1},
                                      gradgobjcoefficients::Array{Float64,1},
                                      solver::AbstractMathProgSolver,
                                      operator::AbstractString;
                                      c1::Float64=1.0,
                                      c2::Float64=1.0,
                                      c3::Float64=1.0,
                                      supportlowerbounds=[-Inf],
                                      supportupperbounds=[Inf])

    # get primary objects
    points = sample.support
    weights = vec(sample.weights)
    n = length(weights)
    ## reset the support bounds if needed
    if isa(supportlowerbounds, Number)
        supportlowerbounds = [supportlowerbounds]
    end
    if isa(supportupperbounds, Number)
        supportupperbounds = [supportupperbounds]
    end
    ## Find spanner edges (consecutive pairs)
    println("[Computing spanner edges]")
    tic(); edges = getspanneredges(points); edgetime = toc()
    println("\tusing $(size(edges,1)) of $(binomial(n,2)) edges")
    tic()
    # start setting up the model
    m = Model(solver=solver)
    # Define variables and specify single variable bounds
    @variable(m, -c1 <= g[1:n] <= c1)
    @variable(m, -c2 <= gprime[1:n] <= c2)
    # Introduce classical Stein program slack variables
    @variable(m, -Inf <= tb[1:(n-1)] <= Inf)
    @variable(m, -Inf <= tu[1:(n-1)] <= Inf)
    # set objective
    gobj = AffExpr(g[1:n], gobjcoefficients, 0.0)
    gprimeobj = AffExpr(gprime[1:n], gradgobjcoefficients, 0.0)
    @objective(m, Max, gobj + gprimeobj)
    # add gprime constraints
    xdistances = points[2:n, 1] - points[1:(n-1), 1]
    for i=1:(n-1)
        @constraint(m, gprime[i] - gprime[i+1] <= c3*xdistances[i])
        @constraint(m, gprime[i] - gprime[i+1] >= -c3*xdistances[i])
    end
    # Introduce constraints tb >= Lb and tu >= Lu:
    # tb_i >= c3/2 * (x_{i+1} - x_i) - (g'(x_i) + g'(x_{i+1}))/2 - c2
    # tu_i >= c3/2 * (x_{i+1} - x_i) + (g'(x_i) + g'(x_{i+1}))/2 - c2
    slackoffsets = (c3/2) * xdistances - c2
    for i=1:(n-1)
        @constraint(m, tb[i] + 0.5 * gprime[i] + 0.5 * gprime[i+1] >= slackoffsets[i])
        @constraint(m, tu[i] - 0.5 * gprime[i] - 0.5 * gprime[i+1] >= slackoffsets[i])
    end
    # Introduce quadratic buffer constraints:
    # |g(x_i)| <= c_1 - (1/(2c_3)) g'(x_i)^2
    for i=1:n
        qexp = QuadExpr([gprime[i]], [gprime[i]], [1/(2*c3)], AffExpr([g[i]], [1.0], 0.0))
        @constraint(m, qexp <= c1)
        qexp = QuadExpr([gprime[i]], [gprime[i]], [1/(2*c3)], AffExpr([g[i]], [-1.0], 0.0))
        @constraint(m, qexp <= c1)
    end
    # Add sharp constraints linking g and g':
    # g(x_{i+1}) - g(x_i) + (g'(x_{i+1}) - g'(x_i))^2/(4c3) - (x_{i+1} - x_i)*(g'(x_i) + g'(x_{i+1}))/2 + (1/c3)(L_u)^2_+ <= (c3/4)(x_{i+1} - x_i)^2
    # g(x_i) - g(x_{i+1}) + (g'(x_{i+1}) - g'(x_i))^2/(4c3) + (x_{i+1} - x_i)*(g'(x_i) + g'(x_{i+1}))/2 + (1/c3)(L_b)^2_+ <= (c3/4)(x_{i+1} - x_i)^2
    scaledsquaredxdistances = (xdistances.^2) .* (c3/4)
    for i=1:(n-1)
        rexp = AffExpr(
            [g[i:(i+1)]; gprime[i:(i+1)]],
            [1.0, -1.0,  xdistances[i]/2, xdistances[i]/2],
            0.0
        )
        qbexp = QuadExpr(
            [gprime[[i, i, i+1]]; tb[i]],
            [gprime[[i, i+1, i+1]]; tb[i]],
            [1.0, -2.0, 1.0, 4.0] ./ (4*c3),
            rexp
        )
        quexp = QuadExpr(
            [gprime[[i, i, i+1]]; tu[i]],
            [gprime[[i, i+1, i+1]]; tu[i]],
            [1.0, -2.0, 1.0, 4.0] ./ (4*c3),
            -1 * rexp
        )

        @constraint(m, qbexp <= scaledsquaredxdistances[i])
        @constraint(m, quexp <= scaledsquaredxdistances[i])
    end

    # get boundary of support
    alpha = supportlowerbounds[1]
    beta = supportupperbounds[1]
    # add support slack constraints if necessary
    if isfinite(alpha)
        xdistance = points[1,1] - alpha
        # |g(x_1)| <= c2 (x_1 - alpha)
        @constraint(m, g[1] <= c2*xdistance)
        @constraint(m, g[1] >= -c2*xdistance)
        # g(x_1) <= g'(x_1)*(x_1 - alpha) + (c3/2)*(x_1-alpha)^2 - 1/(2c3) max{g'(x_1) + c3(x_1 - alpha) - c2, 0}^2,
        # g(x_1) >= g'(x_1)*(x_1 - alpha) - (c3/2)*(x_1-alpha)^2 + 1/(2c3) max{-g'(x_1) + c3(x_1 - alpha) - c2, 0}^2
        @variable(m, -Inf <= alphab <= Inf)
        @variable(m, -Inf <= alphau <= Inf)
        @constraint(m, alphab - gprime[1] >= c3 * xdistance - c2)
        @constraint(m, alphau + gprime[1] >= c3 * xdistance - c2)
        rexp = AffExpr(
            [g[1], gprime[1]],
            [1.0, -xdistance],
            0.0
        )
        qbexp = QuadExpr([alphab], [alphab], [1/(2*c3)], rexp)
        quexp = QuadExpr([alphau], [alphau], [1/(2*c3)], -1 * rexp)
        @constraint(m, qbexp <= (c3/2) * xdistance^2)
        @constraint(m, quexp <= (c3/2) * xdistance^2)
    end
    if isfinite(beta)
        xdistance = beta - points[end,1]
        # |g(x_n)| <= c2 (beta - x_n)
        @constraint(m, g[n] <= c2*xdistance)
        @constraint(m, g[n] >= -c2*xdistance)
        # -g(x_n) <= g'(x_n)*(beta-x_n) + (c3/2)*(beta-x_n)^2 - 1/(2c3) max{g'(x_n) + c3(beta-x_n) - c2, 0}^2
        # -g(x_n) >= g'(x_n)*(beta-x_n) - (c3/2)*(beta-x_n)^2 + 1/(2c3) max{-g'(x_n) + c3(beta-x_n) - c2, 0}^2
        @variable(m, -Inf <= betab <= Inf)
        @variable(m, -Inf <= betau <= Inf)
        @constraint(m, betab + gprime[n] >= c3 * xdistance - c2)
        @constraint(m, betau - gprime[n] >= c3 * xdistance - c2)
        rexp = AffExpr(
            [g[n], gprime[n]],
            [1.0, xdistance],
            0.0
        )
        qbexp = QuadExpr([betab], [betab], [1/(2*c3)], rexp)
        quexp = QuadExpr([betau], [betau], [1/(2*c3)], -1 * rexp)
        @constraint(m, qbexp <= (c3/2) * xdistance^2)
        @constraint(m, quexp <= (c3/2) * xdistance^2)
    end
    # Solve the optimization program
    @time status = JuMP.solve(m)
    solvetime = [toc()]
    # Package the results
    objval = [getobjectivevalue(m)]
    gopt = getvalue(g)[1:n]
    gprimeopt = getvalue(gprime)[1:n]
    operatorgopt = (gopt .* gobjcoefficients) + (gprimeopt .* gradgobjcoefficients)

    AffineDiscrepancyResult(
        points,
        weights,
        edges,
        objval,
        gopt,
        gprimeopt,
        operatorgopt,
        edgetime,
        solvetime,
        operator
    )
end
