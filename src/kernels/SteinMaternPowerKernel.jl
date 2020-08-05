# The SteinMaternPowerKernel is given by
#
# k(x, y) = ((1 + ||x||^alpha)*(1 + ||y||^alpha))^-1 * km(x,y)
#
# where km is the matern kernel.

mutable struct SteinMaternPowerKernel <: SteinMaternWeightedKernel
    # alpha is the exponent on the norms
    alpha::Float64
    # gamma is the rate parameter for the matern kernel
    gamma::Float64
end

# default is use gamma = sqrt{3} and alpha = 2.5
SteinMaternPowerKernel() = SteinMaternPowerKernel(2.5, sqrt(3.0))

# default is use gamma sqrt{3}
SteinMaternPowerKernel(alpha::Float64) = SteinMaternPowerKernel(alpha, sqrt(3.0))

# The term (1 + ||z||^alpha)^-1
function _Qz(ker::SteinMaternPowerKernel, z::Array{Float64, 1})
    Qz = 1.0 + norm(z,2)^ker.alpha
    1.0 / Qz
end

function Q(ker::SteinMaternPowerKernel, x::Array{Float64,1}, y::Array{Float64,1})
    _Qz(ker, x) * _Qz(ker, y)
end

function gradxQ(ker::SteinMaternPowerKernel, x::Array{Float64,1}, y::Array{Float64,1})
    fx = _Qz(ker, x)
    fy = _Qz(ker, y)
    mx = norm(x, 2)^(ker.alpha - 2)
    -x * ker.alpha * fx^2 * fy * mx
end

function gradxyQ(ker::SteinMaternPowerKernel, x::Array{Float64,1}, y::Array{Float64,1})
    mx = norm(x, 2)^(ker.alpha - 2.0)
    my = norm(y, 2)^(ker.alpha - 2.0)
    dot(x, y) * ker.alpha^2 * mx * my * Q(ker, x, y)^2
end
