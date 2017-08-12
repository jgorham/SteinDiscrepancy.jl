# constant used for computation
EPSSCALE = 10

# Computes the Wasserstein distance between a univariate discrete distribution
# and a univariate distribution with an implemented function 'cdf'.
#
# Args:
#   Q - univariate SteinDiscrete distribution; support of Q must be sorted
#   targetcdf - CDF of the (other) univariate distribution P
#
# Returns:
#   Tuple of Wasserstein distance and upper bound on numerical integration
#   error.
function wasserstein1d(Q::SteinDiscrete,
                       targetcdf::Function;
                       lower=-Inf,
                       upper=Inf)
    X = Q.support;
    if size(X,2) != 1
        error("function only defined by univariate distributions")
    end
    if !issorted(X)
        error("Support of Q must be sorted")
    end
    q = Q.weights;
    n = length(q);

    # Compute \int_t |Q(t) - P(t)| dt where Q(t) is the cdf of Q and P(t)
    # is the cdf of P
    # First compute contribution of the interval (lower, x_1)
    # We need to be watchful for machine precision issues
    (integral, max_error) = quadgk(t->thresholdpos(targetcdf(t), EPSSCALE*eps(0.0)), lower, X[1])
    # Cumulative Q weight
    weight = 0;
    for i in 1:(n-1)
        # Compute integral contributed by each (x_i, x_{i+1}] interval
        weight = weight + q[i];
        (estimate, error) = quadgk(t->thresholdpos(abs(targetcdf(t) - weight), EPSSCALE*eps(weight)), X[i], X[i+1]);
        integral = integral + estimate;
        max_error = max_error + error;
    end
    # Add contribution of (x_n, upper)
    (estimate, error) = quadgk(t->thresholdpos(1-targetcdf(t), EPSSCALE*eps(-1.0)), X[n], upper)
    integral = integral + estimate;
    max_error = max_error + error;
    (integral, max_error)
end

# Computes the Wasserstein distance between a univariate discrete distribution
# (represented by points and weights) and a univariate distribution with an
# implemented function 'cdf'.
#
# Args:
#   points - n x 1 array of support points
#   weights - n x 1 array of real-valued weights associated with support points
#   target - univariate SteinDistribution implementing the function 'cdf'
#
# Returns:
#   Tuple of Wasserstein distance and upper bound on numerical integration
#   error.
function wasserstein1d(points::Array{Float64};
                       weights=fill(1/size(points,1), size(points,1)),
                       targetcdf=nothing)
    isa(targetcdf, Function) ||
        error("targetcdf must be the cdf of the continuous distribution.")
    wasserstein1d(SteinDiscrete(points, weights), targetcdf)
end

# This is a helper function that used to avoid weird
# machine precision issues when integrating the cdf.
function thresholdpos(x::Float64, threshold::Float64)
    (x <= threshold) ? 0.0 : x
end

# This function computes the earth movers distance between two
# discrete distributions.
#
# It solves a large LP in order to compute the EMD.
function wassersteindiscrete{T<:Number}(;
    xpoints::AbstractArray{T,2}=[],
    xweights::Array{Float64,1}=fill(1.0/size(xpoints,1), size(xpoints,1)),
    ypoints::AbstractArray{T,2}=[],
    yweights::Array{Float64,1}=fill(1.0/size(ypoints,1), size(ypoints,1)),
    solver=nothing)
    # make sure samples are same dimensionality
    @assert size(xpoints, 2) == size(ypoints, 2)
    # make sure solver is defined
    if isa(solver, AbstractString)
        solver = getsolver(solver)
    end
    isa(solver, AbstractMathProgSolver) ||
        error("Must specify solver of type String or AbstractMathProgSolver")
    # setup discrete distributions
    xdist = SteinDiscrete(xpoints, xweights)
    nx = size(xdist.support, 1)
    ydist = SteinDiscrete(ypoints, yweights)
    ny = size(ydist.support, 1)
    # prepare the manhattan extension graph
    complete_support = [xdist.support; ydist.support]
    graph_support, edgeset = makel1spanner(complete_support)
    # compute the supply for the network problem
    ntotal = size(graph_support,1)
    qx = zeros(ntotal)
    qx[1:nx] = xdist.weights
    qy = zeros(ntotal)
    qy[(nx+1):(nx+ny)] = ydist.weights
    # now solve the min-cost max-flow problem
    (emd, flow, status) = mincostflow(graph_support,
                                      edgeset,
                                      qx,
                                      qy,
                                      solver)
    return (emd, size(graph_support,1), length(edgeset), status)
end

# This function computes the earth movers distance between two discrete
# univariate distributions.
# This is simply int |F1(z) - F2(z)| dz.
function wasserstein1d{T<:Number}(;
    xpoints::AbstractArray{T,1}=[],
    xweights::Array{Float64,1}=fill(1.0/length(xpoints), length(xpoints)),
    ypoints::AbstractArray{T,1}=[],
    yweights::Array{Float64,1}=fill(1.0/size(ypoints,1), size(ypoints,1)))
    # setup discrete distributions
    xdist = SteinDiscrete(xpoints, xweights)
    ydist = SteinDiscrete(ypoints, yweights)
    # now grab the goodies
    xsupp = xdist.support
    qx = xdist.weights
    ysupp = ydist.support
    qy = ydist.weights
    # initialize the indices we walk up the vectors
    ix = 0; iy = 0
    wasserstein = 0.0
    Fx = 0.0
    Fy = 0.0
    currval = -Inf
    if xsupp[1] <= ysupp[1]
        ix = 1
        Fx = qx[1]
        currval = xsupp[1]
    end
    if xsupp[1] >= ysupp[1]
        iy = 1
        Fy = qy[1]
        currval = ysupp[1]
    end
    while ix != length(xsupp) || iy != length(ysupp)
        # handle edge case when we've cycled through the distribution
        if ix == length(xsupp)
            iy += 1
            wasserstein += (ysupp[iy] - currval) * (1.0 - Fy)
            Fy += qy[iy]
            currval = ysupp[iy]
            continue
        end
        if iy == length(ysupp)
            ix += 1
            wasserstein += (xsupp[ix] - currval) * (1.0 - Fx)
            Fx += qx[ix]
            currval = xsupp[ix]
            continue
        end
        # now do the look ahead
        nextval = min(xsupp[ix+1], ysupp[iy+1])
        wasserstein += (nextval - currval) * abs(Fx - Fy)
        if xsupp[ix+1] <= ysupp[iy+1]
            ix += 1
            Fx += qx[ix]
            currval = xsupp[ix]
        else
            iy +=1
            Fy += qy[iy]
            currval = ysupp[iy]
        end
    end
    (wasserstein, ix + iy, ix + iy - 1, :Optimal)
end

# This function approximates the wasserstein metric by using two facts:
# 1. We can compute the EMD for two different discrete distributions.
# 2. The empirical wasserstein metric is root{n}-consistent for the
# true wasserstein metric.
function approxwasserstein{T<:Number}(;
    points::AbstractArray{T,2}=[],
    weights::Array{Float64,1}=fill(1.0/size(points,1), size(points,1)),
    targetsamplegen=nothing,
    replicates::Int=30,
    alpha::Float64=0.95,
    solver=nothing)
    # make sure targetsamplegen is given
    isa(targetsamplegen, Function) ||
        error("Must supply targetsamplegen as a function to generate random samples.")
    # compute a few estimates of wasserstein metric
    wassersteinestimates = Array{Float64}(replicates)
    for ii in 1:replicates
        sample = targetsamplegen()
        (emd, numnodes, numedges, status) =
            wassersteindiscrete(xpoints=points,
                                xweights=weights,
                                ypoints=sample,
                                solver=solver)
        wassersteinestimates[ii] = emd
    end
    # compute CI
    noise = Distributions.TDist(replicates - 1)
    wassmean = mean(wassersteinestimates)
    wassstd = std(wassersteinestimates) / sqrt(replicates)
    zalpha = abs(Distributions.quantile(noise, (1.0 - alpha) / 2))
    return (wassmean - zalpha * wassstd, wassmean + zalpha * wassstd)
end
