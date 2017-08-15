# Computes classical Stein discrepancy based on the Langevin operator
# for univariate target distributions.
# Args:
#  points - n x d array of sample points
#  weights - n x 1 array of real-valued weights associated with sample points
#    (default: equal weights)
#  gradlogdensity - the gradlogdensity of the target distribution
#  solver - optimization program solver supported by JuMP that can solve
#   a quadratically constrained quadratic program (QCQP)
function langevin_classical_discrepancy(points=[],
                                        weights=fill(1/size(points,1), size(points,1)),
                                        gradlogdensity=nothing,
                                        solver=nothing,
                                        kwargs...)
    # we require that the solver is Gurobi or else the quadratic expressions
    # don't work
    try
        isa(solver, Main.Gurobi.GurobiSolver) ||
            error("The solver used must be Gurobi for classical discrepancies.")
    catch
        error("Gurobi must be used as the solver to solve the classical discrepancy.")
    end
    # get primary objects
    sample = SteinDiscrete(points, weights)
    points = sample.support
    weights = vec(sample.weights)
    n = length(weights)
    d = size(points,2)
    # check if d is 1
    (d == 1) || error("The classical discrepancy only works for univariate distributions.")
    # Objective coefficients for each g(x_i)
    gradlogdensities = zeros(n, d)
    for i in 1:n
        gradlogdensities[i,:] = gradlogdensity(points[i,:])
    end
    gobjcoefficients = vec(weights .* gradlogdensities)
    gradgobjcoefficients = weights
    # solve the optimization problem
    result = affine_classical_discrepancy(sample,
                                          gobjcoefficients,
                                          gradgobjcoefficients,
                                          solver,
                                          "langevin";
                                          kwargs...)
    result
end
