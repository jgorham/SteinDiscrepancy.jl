# The SteinMaternUnboundedPowerKernel has of the form:
#
# k(x, y) = ((1 + ||x||^alpha)*(1 + ||y||^alpha)) * km(x,y)
#
# where ||.|| is the L2 norm and km is the matern kernel.

mutable struct SteinMaternUnboundedPowerKernel <: SteinMaternWeightedKernel
    # the alpha parameter
    alpha::Float64
    # the gamma parameter
    gamma::Float64
end

# default is use gamma = sqrt{3} and alpha = 2.5
SteinMaternUnboundedPowerKernel() = SteinMaternUnboundedPowerKernel(2.5, sqrt(3.0))

# default is use gamma sqrt{3}
SteinMaternUnboundedPowerKernel(alpha::Float64) = SteinMaternUnboundedPowerKernel(alpha, sqrt(3.0))

# The term (1 + gamma*||z||^alpha)
function _Qz(ker::SteinMaternUnboundedPowerKernel, z::Array{Float64, 1})
    1.0 + norm(z,2)^ker.alpha
end

# The term ((1 + gamma*||x||^alpha)*(1 + gamma*||y||^alpha))^-1
function Q(ker::SteinMaternUnboundedPowerKernel, x::Array{Float64, 1}, y::Array{Float64, 1})
    _Qz(ker, x) * _Qz(ker, y)
end

function gradxQ(ker::SteinMaternUnboundedPowerKernel, x::Array{Float64,1}, y::Array{Float64,1})
    fy = _Qz(ker, y)
    mx = norm(x,2)^(ker.alpha - 2.0)
    ker.alpha * mx * fy * x
end

function gradxyQ(ker::SteinMaternUnboundedPowerKernel, x::Array{Float64,1}, y::Array{Float64,1})
    mx = norm(x,2)^(ker.alpha - 2)
    my = norm(y,2)^(ker.alpha - 2)
    ker.alpha^2 * mx * my * dot(x, y)
end
