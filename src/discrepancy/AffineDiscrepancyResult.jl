# The returned result from a affine discrepancy

type AffineDiscrepancyResult
    # n x p matrix of distinct sample points
    points::Array{Float64}
    # n x 1 vector of weights associated with each sample
    weights::Array{Float64}
    # Defines Lp norm in Lipschitz constraints:
    # |g(x)-g(y)| <= c_1 ||x-y||_{xnorm}
    xnorm::Float64
    # Pairs of sample points over which smoothness constraints enforced
    edges::Array{Int64,2}
    # 1 x p vector of optimal objective values for each optimization problem
    objectivevalue::Array{Float64}
    # n x p matrix of optimal g function values from each optimization problem
    # at each sample point
    g::Array{Float64}
    # n x p x p tensor of optimal grad g values from each optimization problem
    # at each sample point
    gradg::Array{Float64}
    # n x p matrix of Langevin operator applied to optimal g,
    # (T_Pg)(x) = <grad, g(x)> + <g(x),grad log p(x)>, evaluated at each
    # sample point
    operatorg::Array{Float64}
    # Time taken computing spanner edges
    edgetime::Float64
    # Time taken to set up and solve each optimization program
    solvetime::Array{Float64}
    # The flavor of operator used to construct the objective
    operator::AbstractString
    # Internal constructor assigns default value to xnorm
    AffineDiscrepancyResult(points,weights,edges,objectivevalue,g,gradg,
                              operatorg,edgetime,solvetime,operator) = (
        # Assign default value to xnorm
        xnorm = 1;
        new(points,weights,xnorm,edges,objectivevalue,g,gradg,
            operatorg,edgetime,solvetime,operator)
    )
end
