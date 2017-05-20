# This computes the IPM
#
# sup_{g\in G} |E_P g(X) - E_Q g(Z)|
#
# by using a RKHS that contains only mean-zero functions under P.
# Given some base kernel k(x,x'), we construct a new kernel
# via the mapping
#
# k_0(x, x') = <\grad_x, \grad_x' k(x,x')> +
#   <u(x), \grad_x' k(x,x')> + <u(x'), \grad_x k(x,x')> +
#   <u(x), u(x')> k(x,x')
#
# where u(x) = \grad_x log p(x) is the score function.
#
# This is useful, because if H_0 is the RKHS with kernel k_0,
# we can solve the problem
#
# sup_{h\in H_0} \sum q_i h(x_i) - 1/2 ||h||_{H0}^2
#
# explicity with solution h(x) = \sum q_i k_0(x, x_i). This leaves
# the Stein discrepancy at \sum q_i q_j k_0(x_i, x_j).
#
# Args:
#  points  - the support points comprising the sample distribution
#  weights - the weights of the discrete sample
#  gradlogdensity - the gradlogdensity of the target distribution
#  kernel - the base kernel to make RKHS mean zero under target
#  checkpoints - if not empty, expects an array of ints that
#                mark intermediate n to compute the kernel. This is
#                more optimized than calling this function len(checkpoints) times
#  ncores - the number of cores to break the chunks into
function langevin_kernel_discrepancy(points::Array{Float64},
                                     weights::Array{Float64,1},
                                     gradlogdensity::Function;
                                     kernel=nothing,
                                     checkpoints=[],
                                     ncores=nprocs())
    # make sure kernel is defined
    if !isa(kernel, SteinKernel)
        error("Must supply a subtype of SteinKernel")
    end
    # start the timer
    tic()
    ## Extract inputs
    weights = vec(weights)
    n = length(weights)
    d = size(points,2)
    nocheckpoints = (length(checkpoints) == 0)
    # initialize checkpoints and sort
    if nocheckpoints
        sortedcheckpoints = [n]
        checkpointsindex = [1]
    else
        checkpointsindex = sortperm(checkpoints)
        sortedcheckpoints = checkpoints[checkpointsindex]
    end
    # Compute score functions for each observation
    gradlogdensities = compute_gradlogdensities(gradlogdensity, points)
    # chunk up the data in big batches
    datachunks = Any[
        Any[kernel, points, gradlogdensities, weights, sortedcheckpoints, i, ncores]
        for i in 1:ncores
    ]
    # parallelize the work
    worker_discrepancies = pmap((args) -> compute_kernel_sums(args...), datachunks)
    # merge the worker results and reweight discrepancies
    checkpoint_discrepancies = merge_worker_discrepancies(
        worker_discrepancies
    )
    # only renormalize the discrepancies for the non-CSPD kernels
    if !isa(kernel, SteinConditionallySPDKernel)
        checkpoint_discrepancies = renormalize_discrepancies(
            checkpoint_discrepancies,
            weights,
            sortedcheckpoints
        )
    end
    # put the discrepancies back in original order
    discrepancies = Array(Float64, length(checkpoint_discrepancies))
    discrepancies[checkpointsindex] = checkpoint_discrepancies
    # stop the timer
    solvetime = toc()

    if nocheckpoints
        return LangevinKernelResult(
            points,
            weights,
            discrepancies[1],
            solvetime
        );
    else
        return LangevinKernelCheckpointsResult(
            points,
            weights,
            discrepancies,
            checkpoints,
            solvetime
        );
    end
end

# This is a helper function to compute the score functions
# used in the final kernel evaluations.
function compute_gradlogdensities(gradlogdensity::Function,
                                  points::Array{Float64,2})
    n, d = size(points)
    gradlogdensities = @parallel (vcat) for i=1:n
        gradlogdensity(points[i,:])
    end
    # HACK: we need to extend the dim of gradlogp if d is 1
    if d == 1
        gradlogdensities = reshape(gradlogdensities, size(points))
    end
    gradlogdensities
end

# Computes the sum of discrepancies for indices (i,j) s.t.
# i % gap == istart and j >= i.
#
# This method is a bit clever b/c of checkpoints. It returns
# and array the size of checkpoints of the discrepancies
# that fall into each class. That is, suppose
#
# checkpoints = [n1, n2, n]
#
# Then an array of size 3 will be returned, where the first value
# is all (i, j) pairs above but i <= j <= n1. The second value will
# be all (i, j) pairs above but n1 < i <= j <= n2, and so on.
#
# @params
# kernel: the Stein kernel
# points: the xi
# gradlogdensities: the gradient of log densities
# weights: the weights
# checkpoints: SORTED array of n's to compute discrepancy for first n points
# istart: the lower bound
# gap: the gap to jump
# @returns: array of partial kernel sums
function compute_kernel_sums(kernel::SteinKernel,
                             points::Array{Float64,2},
                             gradlogdensities::Array{Float64,2},
                             weights::Array{Float64,1},
                             checkpoints::Array{Int},
                             istart::Int,
                             gap::Int)
    n = length(weights)
    discrepancies = [0.0 for _ in checkpoints]
    for i in istart:gap:n
        dindex = findfirst(checkpoints .>= i)
        if dindex == 0
            # biggest checkpoint smaller than n, no op
            break
        end
        for j in i:n
            if checkpoints[dindex] < j
                # crossed into a new cross section
                dindex += 1
            end
            m = (i == j) ? 1 : 2
            discrepancies[dindex] += m * compute_kernel_pair(
                 kernel,
                 points[i,:],
                 gradlogdensities[i,:],
                 weights[i],
                 points[j,:],
                 gradlogdensities[j,:],
                 weights[j])
        end
    end
    discrepancies
end

# This is a helper function farmed out to each worker
# that computes k(xi, xj) for the Stein kernel.
# @params
# kernel: the Stein kernel
# xi: the point xi
# ui: the score function at xi
# wi: the weight for xi
# xj: the point xj
# uj: the score function at xj
# wj: the weight for xj
function compute_kernel_pair(
        kernel::SteinKernel,
        xi::Array{Float64,1},
        ui::Array{Float64,1},
        wi::Float64,
        xj::Array{Float64,1},
        uj::Array{Float64,1},
        wj::Float64)
    kij = k0(kernel, xi, xj, ui, uj)
    wi * wj * kij
end

# Private helper function to merge discrepancies from workers
function merge_worker_discrepancies(
    worker_discrepancies::Array{Any,1})
    # merge all the workers
    checkpoint_discrepancies = reduce(+, worker_discrepancies)
    checkpoint_discrepancies
end

function renormalize_discrepancies(
    checkpoint_discrepancies::Array{Float64,1},
    weights::Array{Float64,1},
    checkpoints::Array{Int})
    # sum all the pairs
    cumulative_discrepancies = cumsum(checkpoint_discrepancies)
    # renormalize checkpoint weights to sum to 1
    checkpointweights = map(i -> sum(weights[1:i]), checkpoints)
    cumulative_discrepancies = cumulative_discrepancies ./ checkpointweights.^2
    cumulative_discrepancies
end
