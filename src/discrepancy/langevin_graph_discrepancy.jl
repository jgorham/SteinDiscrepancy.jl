# Computes graph Stein discrepancy based on the Langevin operator.
# Args:
#  sample - SteinDiscrete object representing a sample
#  gradlogdensity - the gradlogdensity of the target distribution
#  solver - optimization program solver supported by JuMP
function langevin_graph_discrepancy(sample::SteinDiscrete,
                                    gradlogdensity::Function;
                                    solver=nothing,
                                    kwargs...)
    # make sure solver is defined
    if isa(solver, AbstractString)
        solver = getsolver(solver)
    end
    isa(solver, AbstractMathProgSolver) ||
        error("Must specify solver of type String or AbstractMathProgSolver")
    ## Extract inputs
    points = sample.support
    weights = sample.weights
    n = length(weights)
    d = size(points,2)
    # Objective coefficients for each g(x_i)
    gradlogdensities = zeros(n, d)
    for i in 1:n
        gradlogdensities[i,:] = gradlogdensity(points[i,:])
    end
    gobjcoefficients = broadcast(*, gradlogdensities, weights)
    # Objective coefficients for each grad g(x_i)
    gradgobjcoefficients = zeros(n, d, d)
    for j = 1:d
        gradgobjcoefficients[:,j,j] = weights
    end
    # solve the optimization problem
    result = affine_graph_discrepancy(sample,
                                      gobjcoefficients,
                                      gradgobjcoefficients,
                                      solver,
                                      "langevin";
                                      kwargs...)
    result
end
