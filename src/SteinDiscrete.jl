# Discrete Measure
#
# Represents a generic discrete distribution with finitely many atoms

# Returns a weighted sample with one row for each distinct row in X and an
# associated weight equaling the sum of all weights.  Rows are sorted in
# ascending order if X is univariate.
function compresssample(X::AbstractArray{T}, q::AbstractArray{Float64}) where T
    n = size(X,1);
    p = size(X,2);
    # Add each row of X with its weight to an accumulator
    a = DataStructures.Accumulator{Array{T,2},Float64}()
    for i in 1:n
        push!(a,X[i:i,:],q[i])
    end
    # Collect distinct points
    nnew = length(a);
    Xnew = zeros(Float64,nnew,p);
    qnew = zeros(Float64,nnew,1);
    for (i,x) in enumerate(keys(a))
        Xnew[i,:] = x;
        qnew[i] = a[x];
    end
    # Sort points if univariate
    if p == 1
        order = sortperm(Xnew[:,1]);
        Xnew = Xnew[order,:];
        qnew = qnew[order,:];
    end
    (Xnew, qnew)
end

mutable struct SteinDiscrete
    # n x p matrix of support points
    support::Array{Float64,2}
    # n x 1 vector of weights associated with support points
    weights::Array{Float64,2}
    # Internal constructor collapses X to its unique sorted elements and
    # adjusts q accordingly; sorts X if univariate.
    SteinDiscrete(support, weights) = (
        (support_,weights_) = compresssample(support,weights);
        return new(support_, weights_);
    )
end

# Constructor with uniform weights q_i = 1/n
SteinDiscrete(X) = SteinDiscrete(
    X,
    [1/size(X,1) for i=1:size(X,1)]
);
