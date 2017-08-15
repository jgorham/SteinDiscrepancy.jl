# Computes graph Stein discrepancy based on the affine objective and a greedy
# 2-spanner of the complete graph on sample points x(i,:) with edges weighted by
# ||x(i,:) - x(i2,:)||
#
# For each j in {1,...,d}, returns a solution g_j to
#   max_{g_j}
#   sum_i q(i) gobjcoeff(i,j) * g_j(x(i,:)) + <gradgobjcoeff(i,:,j), grad g_j(x(i,:)))>
#   subject to, for each spanner edge (i, i2),
#    |g_j(x(i,:))| <= c_{1}, ||grad g_j(x(i,:))||^* <= c_{2},
#    |g_j(x(i,:)) - g_j(x(i2,:))| <= c_{2} ||x(i,:) - x(i2,:)||,
#    ||grad g_j(x(i,:)) - grad g_j(x(i2,:))||^*
#      <= c_3 ||x(i,:) - x(i2,:)||
#    |g_j(x(i,:)) - g_j(x(i2,:)) - <grad g_j(x(i2,:)), x(i,:) - x(i2,:)>|
#      <= (c_3/2) (||x(i,:) - x(i2,:)||)^2
#    |g_j(x(i,:)) - g_j(x(i2,:)) - <grad g_j(x(i,:)), x(i,:) - x(i2,:)>|
#      <= (c_3/2) (||x(i,:) - x(i2,:)||)^2
# where the norm ||.|| is the L1 norm and ||.||^* is its dual.
#
# Args:
#  sample - SteinDiscrete object representing a sample
#  gobjcoefficients - a n x d matrix so jth column has coefficients of
#      the objective for gj
#  gradgobjcoefficients - a n x d x d tensor so [:,:,j] represents
#      the coefficients of grad gj on each row
#  solver - optimization program solver supported by JuMP. Examples:

function affine_graph_discrepancy(sample::SteinDiscrete,
                                  gobjcoefficients::Array{Float64,2},
                                  gradgobjcoefficients::Array{Float64,3},
                                  solver::AbstractMathProgSolver,
                                  operator::AbstractString;
                                  c1::Float64=1.0,
                                  c2::Float64=1.0,
                                  c3::Float64=1.0,
                                  supportlowerbounds=nothing,
                                  supportupperbounds=nothing)

    ## Extract inputs
    points = sample.support
    weights = sample.weights
    n = length(weights)
    d = size(points,2)

    ## reset the support bounds if needed
    if supportlowerbounds == nothing
        supportlowerbounds = repmat([-Inf], d)
    elseif isa(supportlowerbounds, Number)
        supportlowerbounds = rempat([supportlowerbounds], d)
    end
    if supportupperbounds == nothing
        supportupperbounds = repmat([Inf], d)
    elseif isa(supportupperbounds, Number)
        supportupperbounds = rempat([supportupperbounds], d)
    end

    ## Find spanner edge set
    println("[Computing spanner edges]")
    tic(); edges = getspanneredges(points); edgetime = toc()
    println("\tusing $(size(edges,1)) of $(binomial(n,2)) edges");
    if n > 1
        # Compute differences between points connected by an edge
        diffs = points[edges[:,1],:] - points[edges[:,2],:]
        # Compute L1 distances between points connected by an edge
        distances = sum(abs.(diffs),2)
        scaled_squared_distances = (c3/2)*distances.^2
    end

    ## Prepare return values
    objval = Array{Float64}(1, d)
    gopt = Array{Float64}(n, d)
    gradgopt = Array{Float64}(n, d, d)
    solvetime = Array{Float64}(1, d)

    # Distance cutoff for enforcing Lipschitz function constraints
    lipfunccutoff = 2*c1/c2
    # Distance cutoff for enforcing Lipschitz gradient constraints
    lipgradcutoff = 2*c2/c3
    # Distance cutoff for enforcing Taylor compatibility constraints
    taylorcutoff = 4*c2/c3

    ## Solve a different problem for each sample coordinate
    println("[Solving optimization program]")
    for j = 1:d
        tic()
        ## Define optimization problem
        m = Model(solver=solver)
        # Define optimization variables
        @variable(m, -c1 <= g[i=1:n] <= c1)
        @variable(m, -c2 <= gradg[i=1:n,k=1:d] <= c2)
        # Define the optimization objective
        gobj = AffExpr(g[1:n], vec(gobjcoefficients[:,j]), 0.0)
        gradgobj = AffExpr()
        for k = 1:d
            gradgobj += AffExpr(gradg[1:n,k], vec(gradgobjcoefficients[1:n,k,j]), 0.0)
        end
        @objective(m, Max, gobj + gradgobj)
        # Find finite limits of support in dimension j
        limits = [
            supportlowerbounds[j],
            supportupperbounds[j]
        ]
        limits = filter(isfinite, limits)
        # Add boundary constraints if needed
        for i = 1:n, bj = limits
            slackij = points[i,j] - bj
            # Add constraints to ensure gj can vanish on boundary
            # whenever abs(slackij) < lipfunccutoff
            # (otherwise constraints will never be active)
            if abs(slackij) < lipfunccutoff
                @constraint(m, g[i] >= -c2 * abs(slackij))
                @constraint(m, g[i] <= c2 * abs(slackij))
            end
            # Add constraints to ensure \grad gj can vanish in non-j dimension
            # whenever abs(slackij) < lipgradcutoff
            # (otherwise constraints will never be active)
            if abs(slackij) < lipgradcutoff
                constrained_dims = filter(x -> x != j, 1:d)
                for k = constrained_dims
                    @constraint(m, gradg[i,k] >= -c3 * abs(slackij))
                    @constraint(m, gradg[i,k] <= c3 * abs(slackij))
                end
            end
            # Add \grad gj constraints on jth dimension
            # whenever abs(slackij) < taylorcutoff
            # (otherwise constraints will never be active)
            if abs(slackij) < taylorcutoff
                @constraint(m, g[i] - slackij * gradg[i,j] <= (c3/2) * slackij^2)
                @constraint(m, g[i] - slackij * gradg[i,j] >= (-c3/2) * slackij^2)
            end
        end
        # Add pairwise constraints
        if n > 1
            for i = 1:length(distances)
                v1 = edges[i,1]; v2 = edges[i,2]
                # Add Lipschitz function constraints
                # whenever distance < lipgradcutoff
                # (otherwise constraints will never be active)
                if distances[i] < lipfunccutoff
                    @constraint(m, g[v1] - g[v2] <= c2 * distances[i])
                    @constraint(m, g[v1] - g[v2] >= -c2 * distances[i])
                end
                # Add Lipschitz gradient constraints
                # whenever distances[i] < lipgradcutoff
                # (otherwise constraints will never be active)
                if distances[i] < lipgradcutoff
                    for k = 1:d
                        @constraint(m, gradg[v1,k] - gradg[v2,k] <= c3*distances[i])
                        @constraint(m, gradg[v1,k] - gradg[v2,k] >= -c3*distances[i])
                    end
                end
                # Add Taylor compatibility constraints relating g and gradg
                # whenever distance < taylorcutoff
                # (otherwise constraints will never be active)
                if distances[i] < taylorcutoff
                    expr1 = AffExpr(vec(gradg[v1,:]), vec(diffs[i,:]), 0.0)
                    @constraint(m, g[v1] - g[v2] - expr1 <= scaled_squared_distances[i])
                    @constraint(m, g[v1] - g[v2] - expr1 >= -scaled_squared_distances[i])
                    expr2 = AffExpr(vec(gradg[v2,:]), vec(diffs[i,:]), 0.0)
                    @constraint(m, g[v1] - g[v2] - expr2 <= scaled_squared_distances[i])
                    @constraint(m, g[v1] - g[v2] - expr2 >= -scaled_squared_distances[i])
                end
            end
        end
        # Solve the problem
        @time status = JuMP.solve(m)
        solvetime[j] = toc()
        # Package the results
        objval[j] = getobjectivevalue(m)
        gopt[:,j] = getvalue(g)[1:n]
        gradgopt[:,:,j] = getvalue(gradg)[1:n, 1:d]
    end
    # Compute (T_Pg)(x) = {g part} + {grad g part}
    operatorgopt = sum(gopt .* gobjcoefficients,2)
    for j = 1:d
        operatorgopt += sum(gradgopt[:,:,j] .* gradgobjcoefficients[:,:,j],2)
    end

    AffineDiscrepancyResult(
        points,
        weights,
        edges,
        objval,
        gopt,
        gradgopt,
        operatorgopt,
        edgetime,
        solvetime,
        operator
    );
end

