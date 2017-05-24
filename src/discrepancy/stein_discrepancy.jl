# Computes Stein discrepancy between a weighted sample and a target with gradlogdensity
#
# Args:
#   points - n x d array of sample points
#   weights - n x 1 array of real-valued weights associated with sample points
#     (default: equal weights)
#   gradlogdensity - a function taking point x in Rd and outputting grad log p(x)
#   operator - string in {"langevin", "riemannian-langevin"} indicating the Stein operator
#     to use (default: "langevin")
#   method - One of {"graph", "classical", "kernel"}, this uses different
#     methods in order to construct the different discrepancies.

function stein_discrepancy(; points=[],
                           gradlogdensity=nothing,
                           method="graph",
                           operator="langevin",
                           kwargs...)
    # Check arguments
    isempty(points) && error("Must provide non-empty array of sample points")
    isa(gradlogdensity, Function) ||
        error("Must specify gradlogdensity of type Function")

    if method in ["graph", "classical"]
        # call appropriate graph discrepancy
        if method == "graph"
            if operator == "langevin"
                langevin_graph_discrepancy(;points=points, gradlogdensity=gradlogdensity, kwargs...)
            elseif operator == "riemannian-langevin"
                riemannian_langevin_graph_discrepancy(;points=points, gradlogdensity=gradlogdensity, kwargs...)
            else
                error("unrecognized operator: $(operator)")
            end
        elseif method == "classical"
            if operator == "langevin"
                langevin_classical_discrepancy(;points=points, gradlogdensity=gradlogdensity, kwargs...)
            else
                error("unrecognized operator: $(operator)")
            end
        end
    elseif method == "kernel"
        if operator == "langevin"
            langevin_kernel_discrepancy(;points=points, gradlogdensity=gradlogdensity, kwargs...)
        else
            error("unrecognized operator: $(operator)")
        end
    else
        error("unrecognized method: $(method)")
    end
end

# stein_discrepancy aliases
function ksd(;kwargs...)
    stein_discrepancy(; method="kernel", kwargs...)
end

function gsd(;kwargs...)
    stein_discrepancy(; method="graph", kwargs...)
end
