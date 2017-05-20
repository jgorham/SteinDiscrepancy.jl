# The SteinGaussianRectangularPowerKernel has base kernel of the form:
#
# k(x, y) = \prod_{j: l[j] > -inf} (x_j - l_j)(y_j - l_j) *
#           \prod_{k: u[k] <  inf} (x_k - u_k)(y_k - u_k) *
#           exp(-1*||x-y||^2/(2*beta))
#
# on the domain [l_1, u_1] x ... x [l_d, u_d].

type SteinGaussianRectangularDomainKernel <: SteinGaussianWeightedKernel
    # the beta parameter
    beta::Float64
    # the lower bound vector (l_1, ..., l_d)
    l::Array{Float64,1}
    # the upper bound vector (u_1, ..., u_d)
    u::Array{Float64,1}
end

# default beta is 1.0
SteinGaussianRectangularDomainKernel(l::Array{Float64,1}, u::Array{Float64,1}) =
SteinGaussianRectangularDomainKernel(1.0, l, u)

# default domain is [0,1]^d
SteinGaussianRectangularDomainKernel(d::Int64) =
SteinGaussianRectangularDomainKernel(0.0 * ones(d), 1.0 * ones(d))

# domain is [l,u]^d
SteinGaussianRectangularDomainKernel(l::Float64, u::Float64, d::Int64) =
SteinGaussianRectangularDomainKernel(l * ones(d), u * ones(d))

# domain is [l,u] for 1-d
SteinGaussianRectangularDomainKernel(l::Float64, u::Float64) =
SteinGaussianRectangularDomainKernel(l, d, 1)

function Q(kernel::SteinGaussianRectangularDomainKernel,
              x::Array{Float64,1},
              y::Array{Float64,1})
    J = find(kernel.l .> -Inf)
    K = find(kernel.u .< Inf)
    prodl = prod(x[J] - kernel.l[J]) * prod(y[J] - kernel.l[J])
    produ = prod(x[K] - kernel.u[K]) * prod(y[K] - kernel.u[K])
    prodl * produ
end

function dxj_Q(kernel::SteinGaussianRectangularDomainKernel,
                  x::Array{Float64,1},
                  y::Array{Float64,1},
                  j::Int64)
    l = kernel.l; u = kernel.u
    finitel = l[j] > -Inf
    finiteu = u[j] < Inf
    if !finitel && !finiteu
        return 0.0
    end

    fullprod = Q(kernel, x, y)
    if !finitel && finiteu
        fullprod / (x[j] - u[j])
    elseif finitel && !finiteu
        fullprod / (x[j] - l[j])
    else
        fullprod * (2*x[j] - l[j] - u[j]) / ((x[j] - l[j]) * (x[j] - u[j]))
    end
end

function dxjyj_Q(kernel::SteinGaussianRectangularDomainKernel,
                    x::Array{Float64,1},
                    y::Array{Float64,1},
                    j::Int64)
    l = kernel.l; u = kernel.u
    finitel = l[j] > -Inf
    finiteu = u[j] < Inf
    if !finitel && !finiteu
        return 0.0
    end

    fullprod = Q(kernel, x, y)
    if !finitel && finiteu
        fullprod /= ((x[j] - u[j]) * (y[j] - u[j]))
    elseif finitel && !finiteu
        fullprod /= ((x[j] - l[j]) * (y[j] - l[j]))
    else
        fullprod *= ((2*x[j] - l[j] - u[j]) * (2*y[j] - l[j] - u[j]))
        fullprod /= ((x[j] - l[j]) * (x[j] - u[j]) * (y[j] - l[j]) * (y[j] - u[j]))
    end
    fullprod
end
