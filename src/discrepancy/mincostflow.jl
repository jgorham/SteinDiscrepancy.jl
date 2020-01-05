# Min cost max flow

# This solves the min-cost max flow problem using an LP.
# There are far more efficient algorithms out there (e.g. network
# simplex) but most implementations are only for floats.
#
# Args:
#   X - n x p matrix of points in R^d
#   E - the edge set for the rows of X
#   q1 - the pmf for the source distribution
#   q2 - the pmf for the target distribution
#   solver - the LP solver used to compute the flow.
# Returns
#   mincost - the minimum cost of the max flow
#   flow - the flows to achieve the min cost
#   status - the status of the LP solver.

function mincostflow(X::AbstractArray{T,2},
                     E::Array{Tuple{Int,Int},1},
                     q1::AbstractArray{T,1},
                     q2::AbstractArray{T,1},
                     solver::AbstractMathProgSolver) where {T<:Number}
    # number of points
    (n, d) = size(X)
    # size of supports
    numedges = length(E)
    # construct distances
    dists = zeros(numedges)
    for ii = 1:numedges
        dists[ii] = norm(X[E[ii][1],:] - X[E[ii][2],:], 1)
    end
    # setup model
    m = Model(solver=solver)
    # flow on edges, twice the number for both directions
    @variable(m, f[i=1:(2*numedges)] >= 0)
    # construct objective
    costobjective = f[1:(2*numedges)]'[dists; dists]
    @objective(m, Min, costobjective)
    # first we index vertices by incident edges
    inedges_for_node = Dict{Int,Array{Int,1}}()
    outedges_for_node = Dict{Int,Array{Int,1}}()
    for (edgeid, edge) in enumerate(E)
        (from, to) = edge
        _addedgetonodeindex!(inedges_for_node, outedges_for_node,
                             edgeid, from, to)
        _addedgetonodeindex!(inedges_for_node, outedges_for_node,
                             edgeid + numedges, to, from)
    end
    # impose flow constraints for every pair
    for ii=1:n
        in_edgeids = inedges_for_node[ii]
        out_edgeids = outedges_for_node[ii]
        @constraint(m, sum(f[in_edgeids]) - sum(f[out_edgeids]) == q2[ii] - q1[ii])
    end
    # Solve the problem
    @time status = JuMP.solve(m)
    mincost = getobjectivevalue(m)
    directedflows = getvalue(f)[1:(2*numedges)]
    flows = directedflows[1:numedges] - directedflows[numedges+(1:numedges)]
    # return the values
    return (mincost, flows, status)
end

function _addedgetonodeindex!(
    inedges_for_node::Dict{Int,Array{Int,1}},
    outedges_for_node::Dict{Int,Array{Int,1}},
    edgeid::Int,
    from::Int,
    to::Int)
    # first do the normal orientation
    if !haskey(inedges_for_node, to)
        inedges_for_node[to] = Int[]
    end
    if !haskey(outedges_for_node, from)
        outedges_for_node[from] = Int[]
    end
    push!(inedges_for_node[to], edgeid)
    push!(outedges_for_node[from], edgeid)
end

