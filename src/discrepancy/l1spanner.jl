# 1-Steiner spanner functionality

# Returns a graph G = (V, E) which is a Manhattan network of the original point
# set, i.e., a graph where each edge is parallel to one of the main coordinate
# axes and each pair of points has a path in the graph equal to the L1 distance.
# If the original set S has n points, this graph will have O(n log n) points
# and edges. Based on the algorithm from Gudmundsson et al.
#
# Args:
#   S - n x p matrix of points in R^d
# Returns
#   X - the original points plus the new projected points
#   E - an array of (int, int) representing the edges of the graph where the
#       int index the rows of Xnew.

# constant used for numerical issues
EPS = 10 * eps(0.0)

function makel1spanner(X::AbstractArray{T,2}) where {T<:Number}
    n, d = size(X)
    # copy so we don't corrupt the original data
    Xnew = copy(X)
    E = Tuple{Int, Int}[]
    # begin the recursion!
    Xnew = _makel1spanner!(Xnew, E, collect(1:n), d)
    return (Xnew, E)
end

function _makel1spanner!(X::AbstractArray{T,2},
                         E::Array{Tuple{Int,Int},1},
                         Xidx::Array{Int,1},
                         k::Int) where {T<:Number}
    # initialize the iterative counters
    Xindexs = Array{Int,1}[]
    k_indexs = Int[]
    push!(Xindexs, Xidx)
    push!(k_indexs, k)
    # now iterate through the recursion
    while length(Xindexs) > 0
        # pop off the local counters
        Xidx = pop!(Xindexs)
        k = pop!(k_indexs)
        # this is crazy, but if points already lie on hyperplane,
        # we should decrement k
        while (k > 1) && (maximum(X[Xidx,k]) - minimum(X[Xidx,k])) < EPS
            k -= 1
        end
        # handle base case first
        if k == 0 || length(Xidx) <= 1
            continue
        elseif k == 1
            # in this case, we only must add the adjacent points
            pointidx = sortperm(X[Xidx,1])
            for ii in 1:(length(pointidx)-1)
                push!(E, (Xidx[pointidx[ii]], Xidx[pointidx[ii+1]]))
            end
            continue
        else
            n = size(X,1)
            # make projected points (be sure to copy!)
            P = copy(X[Xidx,:])
            pstar = median(P[:,k])
            P[:,k] = pstar
            # add projected points
            X = vcat(X, P)
            nP = size(P,1)
            # add projected edges
            for ii in 1:nP
                push!(E, (Xidx[ii], n + ii))
            end
            Lhighidx = Xidx[X[Xidx,k] .> pstar]
            Llowidx = Xidx[X[Xidx,k] .< pstar]
            # Make sure we put median points on smaller indices
            Lequalidx = Xidx[X[Xidx,k] .== pstar]
            if length(Lequalidx) > 0 && (length(Llowidx) < length(Lhighidx))
                Llowidx = vcat(Llowidx, Lequalidx)
            elseif length(Lequalidx) > 0
                Lhighidx = vcat(Lhighidx, Lequalidx)
            end
            # call the operation on the upper half of original points
            push!(Xindexs, Lhighidx)
            push!(k_indexs, k)
            # call the operation on the lower half of original points
            push!(Xindexs, Llowidx)
            push!(k_indexs, k)
            # call the operation on the projected points
            Pidx = collect((n + 1):(n + nP))
            push!(Xindexs, Pidx)
            push!(k_indexs, k - 1)
        end
    end
    # return X!
    return X
end
