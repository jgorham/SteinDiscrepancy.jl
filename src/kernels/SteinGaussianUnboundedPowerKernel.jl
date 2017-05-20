# The SteinGaussianUnboundedPowerKernel has base kernel of the form:
#
# k(x, y) = ((1 + gamma*||x||^alpha)*(1 + gamma*||y||^alpha)) *
#   exp(-1/(2*beta)*||x-y||^2)
#
# where ||.|| is the L2 norm. This is gauranteed to enforce
# that k is a universal kernel that vanishes at infinity.

type SteinGaussianUnboundedPowerKernel <: SteinGaussianWeightedKernel
    # the alpha parameter
    alpha::Float64
    # the beta parameter
    beta::Float64
    # the gamma parameter
    gamma::Float64
end

# have default parameters of alpha = 2.5,  beta = 1.0, gamma = 1.0
SteinGaussianUnboundedPowerKernel() = SteinGaussianUnboundedPowerKernel(2.5, 1.0, 1.0)

SteinGaussianUnboundedPowerKernel(alpha::Float64) = SteinGaussianUnboundedPowerKernel(alpha, 1.0, 1.0)

SteinGaussianUnboundedPowerKernel(alpha::Int) = SteinGaussianUnboundedPowerKernel(float(alpha), 1.0, 1.0)

# The term (1 + gamma*||z||^alpha)
function _Qz(ker::SteinGaussianUnboundedPowerKernel, z::Array{Float64, 1})
    1.0 + ker.gamma * norm(z,2)^ker.alpha
end

# The term ((1 + gamma*||x||^alpha)*(1 + gamma*||y||^alpha))^-1
function Q(ker::SteinGaussianUnboundedPowerKernel, x::Array{Float64, 1}, y::Array{Float64, 1})
    _Qz(ker, x) * _Qz(ker, y)
end

function dxj_Q(ker::SteinGaussianUnboundedPowerKernel, x::Array{Float64,1}, y::Array{Float64,1}, j::Int64)
    fy = _Qz(ker, y)
    mx = norm(x,2)^(ker.alpha - 2)
    mx * ker.gamma * ker.alpha * fy * x[j]
end

function dxjyj_Q(ker::SteinGaussianUnboundedPowerKernel, x::Array{Float64,1}, y::Array{Float64,1}, j::Int64)
    mx = norm(x,2)^(ker.alpha - 2)
    my = norm(y,2)^(ker.alpha - 2)

    ker.gamma^2 * ker.alpha^2 * mx * my * x[j] * y[j]
end
