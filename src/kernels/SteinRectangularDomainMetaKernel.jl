# The SteinRectangularDomainMetaKernel is a "meta" kernel of the form:
#
# kR(x, y) = \prod_{j: l[j] > -inf} (x_j - l_j)(y_j - l_j) *
#            \prod_{k: u[k] <  inf} (x_k - u_k)(y_k - u_k) *
#            k(x,y)
#
# on the domain [l_1, u_1] x ... x [l_d, u_d].
# We assume subtypes of this define the fields:
#
# l, u, and basekernel.

mutable struct SteinRectangularDomainMetaKernel <: SteinKernel
    # the base kernel k(x,y)
    basekernel::SteinKernel
    # the lower bound vector (l_1, ..., l_d)
    l::Array{Float64,1}
    # the upper bound vector (u_1, ..., u_d)
    u::Array{Float64,1}
end

# domain is [l,u]^d
SteinRectangularDomainMetaKernel(basekernel::SteinKernel, l::Float64, u::Float64, d::Int64) =
SteinRectangularDomainMetaKernel(basekernel, l * ones(d), u * ones(d))

# HELPER FUNCTIONS

function w(ker::SteinRectangularDomainMetaKernel,
           x::Array{Float64,1})
    J = find(ker.l .> -Inf)
    K = find(ker.u .< Inf)
    prodl = prod(x[J] - ker.l[J])
    produ = prod(ker.u[K] - x[K])
    prodl * produ
end

function w(ker::SteinRectangularDomainMetaKernel,
           x::Array{Float64,1},
           y::Array{Float64,1})
    w(ker, x) * w(ker, y)
end

function alpha(ker::SteinRectangularDomainMetaKernel,
               x::Array{Float64,1})
    l = ker.l; u = ker.u
    d = length(x)
    fullprod = w(ker, x)
    result = zeros(d)

    for j in 1:d
        finitel = l[j] > -Inf
        finiteu = u[j] < Inf
        if !finitel && !finiteu
            continue
        end

        if !finitel && finiteu
            result[j] = fullprod / (x[j] - u[j])
        elseif finitel && !finiteu
            result[j] = fullprod / (x[j] - l[j])
        else
            result[j] = fullprod * (2*x[j] - l[j] - u[j]) / ((x[j] - l[j]) * (x[j] - u[j]))
        end
    end
    result
end

# Definition of kernel

function k(ker::SteinRectangularDomainMetaKernel,
           x::Array{Float64,1},
           y::Array{Float64,1})
    w(ker, x, y) * k(ker.basekernel, x, y)
end

function gradxk(ker::SteinRectangularDomainMetaKernel,
                x::Array{Float64,1},
                y::Array{Float64,1})
    wx = w(ker, x)
    wy = w(ker, y)
    wx * wy * gradxk(ker.basekernel, x, y) + wy * alpha(ker, x) * k(ker.basekernel, x, y)
end

function gradxyk(ker::SteinRectangularDomainMetaKernel,
                 x::Array{Float64,1},
                 y::Array{Float64,1})
    wx = w(ker, x)
    wy = w(ker, y)
    ax = alpha(ker, x)
    ay = alpha(ker, y)
    dot(ax, ay) * k(ker.basekernel, x, y) +
        wx * dot(ay, gradxk(ker.basekernel, x, y)) +
        wy * dot(ax, gradyk(ker.basekernel, x, y)) +
        wx * wy * gradxyk(ker.basekernel, x, y)
end
