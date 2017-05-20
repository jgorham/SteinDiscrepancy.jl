# SteinMaternWeightedKernel
#
# These represent a family of kernels of the form
#
# k(x, y) = Q(x,y) k_0(x,y)
#
# where Q(x,y) is such that k is indeed a valid kernel and
# k_0 is the Matern radial kernel. That is,
#
# k_0(x, y) = (1 + gamma * r) * exp{-gamma * r}
#
# where r = ||x-y||. This is a norm for the order
# 2 Sobolev space H_2. This is the Matern covariance
# associated with v = 3/2.
#
# See http://www.gaussianprocess.org/gpml/chapters/RW4.pdf
# or http://arxiv.org/pdf/1204.6448.pdf for more details.

abstract SteinMaternWeightedKernel <: SteinKernel

### METHODS TO IMPLEMENT

# the function that evaluates Q(x,y)
function Q(ker::SteinMaternWeightedKernel, x::Array{Float64,1}, y::Array{Float64,1})
    error("Must implement the method Q")
end

# the function that implements gradx Q
function gradxQ(ker::SteinMaternWeightedKernel, x::Array{Float64,1}, y::Array{Float64,1})
    error("Must implement the method gradxQ")
end

# the function that implements gradx \dot grady Q
function gradxyQ(ker::SteinMaternWeightedKernel, x::Array{Float64,1}, y::Array{Float64,1})
    error("Must implement the method gradxyQ")
end

### INDUCED METHODS

function _maternk(ker::SteinMaternWeightedKernel, x::Array{Float64,1}, y::Array{Float64,1})
    r = norm(x - y)
    gamma = ker.gamma
    (1 + gamma * r) * exp(-gamma * r)
end

function _materngradxk(ker::SteinMaternWeightedKernel, x::Array{Float64,1}, y::Array{Float64,1})
    r = norm(x - y)
    gamma = ker.gamma
    (gamma^2 * exp(-gamma * r)) .* (y - x)
end

function _materngradyk(ker::SteinMaternWeightedKernel, x::Array{Float64,1}, y::Array{Float64,1})
    _materngradxk(ker, y, x)
end

function _materngradxyk(ker::SteinMaternWeightedKernel, x::Array{Float64,1}, y::Array{Float64,1})
    r = norm(x - y)
    gamma = ker.gamma
    p = length(x)
    gamma^2 * (p - gamma * r) * exp(-gamma * r)
end

function gradyQ(ker::SteinMaternWeightedKernel, x::Array{Float64,1}, y::Array{Float64,1})
    gradxQ(ker, y, x)
end

# defintion of the base kernel
function k(ker::SteinMaternWeightedKernel, x::Array{Float64, 1}, y::Array{Float64, 1})
    Q(ker, x, y) * _maternk(ker, x, y)
end

# implements \grad_x k(x,y)
function gradxk(ker::SteinMaternWeightedKernel, x::Array{Float64, 1}, y::Array{Float64, 1})
    _maternk(ker, x, y) * gradxQ(ker, x, y) + Q(ker, x, y) * _materngradxk(ker, x, y)
end

# implements <\grad_y, \grad_x k(x,y)>
function gradxyk(ker::SteinMaternWeightedKernel, x::Array{Float64, 1}, y::Array{Float64, 1})
    order0 = gradxyQ(ker, x, y) * _maternk(ker, x, y)
    order1 = dot(gradxQ(ker, x, y), _materngradyk(ker, x, y)) +
        dot(gradyQ(ker, x, y), _materngradxk(ker, x, y))
    order2 = Q(ker, x, y) * _materngradxyk(ker, x, y)

    order0 + order1 + order2
end
