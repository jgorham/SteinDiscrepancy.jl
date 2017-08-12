# SteinGaussianWeightedKernel
#
# These represent a family of kernels of the form
#
# k(x, y) = Q(x,y) exp(-||x-y||^2/(2*beta))
#
# where Q(x,y) is such that k is indeed a valid kernel.
# This kernel is combined with the score function u(x) in order
# generate the mean-zero RKHS.

abstract type SteinGaussianWeightedKernel <: SteinKernel end

### METHODS TO IMPLEMENT

# the function that evaluates Q(x,y)
function Q(kernel::SteinGaussianWeightedKernel, x::Array{Float64,1}, y::Array{Float64,1})
    error("Must implement the method Q")
end

# the function that implements d/dxj Q(x,y)
function dxj_Q(kernel::SteinGaussianWeightedKernel, x::Array{Float64,1}, y::Array{Float64,1}, j::Int64)
    error("Must implement the method dxj_Q")
end

# the function that implements d^2/dxjdyj Q(x,y)
function dxjyj_Q(kernel::SteinGaussianWeightedKernel, x::Array{Float64,1}, y::Array{Float64,1}, j::Int64)
    error("Must implement the method dxjyj_Q")
end

### INDUCED METHODS

function gaussian(kernel::SteinGaussianWeightedKernel, x::Array{Float64,1}, y::Array{Float64,1})
    exp(-1 * norm(x-y,2)^2 / (2 * kernel.beta))
end

# the function that implements d/dyj Q(x,y)
# where we assume Q(x,y) = Q(y,x)
function dyj_Q(kernel::SteinGaussianWeightedKernel, x::Array{Float64,1}, y::Array{Float64,1}, j::Int64)
    dxj_Q(kernel, y, x, j)
end

# defintion of the base kernel
function k(kernel::SteinGaussianWeightedKernel, x::Array{Float64, 1}, y::Array{Float64, 1})
    Q(kernel, x, y) * gaussian(kernel, x, y)
end

# implements \grad_x k(x,y) = gaussian(ker, x, y) * [\grad_x Q(x,y) + (y - x) * Q(x,y) / beta]
function gradxk(kernel::SteinGaussianWeightedKernel, x::Array{Float64, 1}, y::Array{Float64, 1})
    d = length(x)
    base = [dxj_Q(kernel, x, y, j) + (y[j] - x[j]) * Q(kernel, x, y) / kernel.beta for j=1:d]
    base .* gaussian(kernel, x, y)
end

# implements <\grad_y, \grad_x k(x,y)>
function gradxyk(kernel::SteinGaussianWeightedKernel, x::Array{Float64, 1}, y::Array{Float64, 1})
    d = length(x)
    Qterm = Q(kernel, x, y)
    gaussianterm = gaussian(kernel, x, y)

    gradxQ = [dxj_Q(kernel, x, y, j) for j=1:d]
    gradyQ = [dyj_Q(kernel, x, y, j) for j=1:d]

    order0 = (d/kernel.beta - norm(x-y,2)^2 / kernel.beta^2) * Qterm * gaussianterm
    order1 = (gaussianterm / kernel.beta) * dot(gradxQ - gradyQ, x-y)
    order2 = gaussianterm * sum([dxjyj_Q(kernel, x, y, j) for j=1:d])

    order0 + order1 + order2
end
