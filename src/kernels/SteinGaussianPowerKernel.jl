# The SteinGaussianPowerKernel has base kernel of the form:
#
# k(x, y) = ((1 + gamma*||x||^alpha)*(1 + gamma*||y||^alpha))^-1 *
#   exp(-1/(2*beta)*||x-y||^2)
#
# where ||.|| is the L2 norm. This is gauranteed to enforce
# that k is a universal kernel that vanishes at infinity.

mutable struct SteinGaussianPowerKernel <: SteinGaussianWeightedKernel
    # the alpha parameter
    alpha::Float64
    # the beta parameter
    beta::Float64
    # the gamma parameter
    gamma::Float64
end

# have default parameters of alpha = 2.5,  beta = 1.0, gamma = 1.0
SteinGaussianPowerKernel() = SteinGaussianPowerKernel(2.5, 1.0, 1.0)

SteinGaussianPowerKernel(alpha::Float64) = SteinGaussianPowerKernel(alpha, 1.0, 1.0)

# The term (1 + gamma*||z||^alpha)^-1
function _Qz(ker::SteinGaussianPowerKernel, z::Array{Float64, 1})
    Qz = 1 + ker.gamma * norm(z,2)^ker.alpha
    1.0 / Qz
end

# The term ((1 + gamma*||x||^alpha)*(1 + gamma*||y||^alpha))^-1
function Q(ker::SteinGaussianPowerKernel, x::Array{Float64, 1}, y::Array{Float64, 1})
    _Qz(ker, x) * _Qz(ker, y)
end

function dxj_Q(ker::SteinGaussianPowerKernel, x::Array{Float64,1}, y::Array{Float64,1}, j::Int64)
    fx = _Qz(ker, x); fy = _Qz(ker, y)
    mx = norm(x, 2)^(ker.alpha - 2)
    -x[j] * ker.alpha * ker.gamma * fx^2 * fy * mx
end

function dxjyj_Q(ker::SteinGaussianPowerKernel, x::Array{Float64,1}, y::Array{Float64,1}, j::Int64)
    mx = norm(x, 2)^(ker.alpha - 2.0)
    my = norm(y, 2)^(ker.alpha - 2.0)
    x[j] * y[j] * ker.alpha^2 * ker.gamma^2 * mx * my * Q(ker, x, y)^2
end
