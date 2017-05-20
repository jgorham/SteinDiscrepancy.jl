# Geometric spanner functionality

# Returns the edges of a geometric t-spanner for a weighted graph with
# vertices representing the rows of X and edge weights given by the
# L1 distance between vertices
#
# Args:
#   X - n x p matrix of points in R^d
#   t - geometric spanner dilation factor (2.0 by default)
#   valid_edges - an array of vertex index pairs representing the initial
#     edges in the graph; if empty, it is assumed that initial graph is
#     the complete graph on X (empty by default)
function getspanneredges(X::Array{Float64};
                         t::Float64=2.0,
                         valid_edges::Array{Int64,2}=Array(Int64,0,0))
    n = size(X, 1)
    # Return empty edge set of only one point in X
    n > 1 || return Array(Int64,0,2)

    p = size(X, 2)
    if p == 1 && isempty(valid_edges)
        # In the univariate complete graph case, return edges between
        # consecutive points in sorted order
        sortorder = sortperm(vec(X))
        return [sortorder[1:n-1] sortorder[2:n]]
    end

    pointlist = vec(X.');
    valid_edges_vec = vec(valid_edges.');
    num_valid_edges = size(valid_edges, 1);
    spannerlib = Libdl.dlopen(libsteinspanner);

    ed = ccall(
        Libdl.dlsym(spannerlib, :prune_edges),
        Ptr{Cint},
        (Ptr{Cdouble}, Cint, Cint, Cdouble, Ptr{Clonglong}, Cint),
        pointlist, n, p, t, valid_edges_vec, num_valid_edges
    );

    numedges = unsafe_load(ed, 1);
    c_edges = Array(Cint, 2*numedges);

    ccall(
        Libdl.dlsym(spannerlib, :load_edges),
        Void,
        (Ptr{Void}, Ptr{Cint}),
        ed, c_edges
    );

    return reshape(c_edges + 1, 2, convert(Int64, numedges)).';
end
