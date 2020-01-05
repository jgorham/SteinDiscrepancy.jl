# See http://arxiv.org/pdf/1204.6448v3.pdf for more details
#
# The SteinMaternTensorizedKernel is given by
#
# k(x, y) = (1/(8 sigma^3)) * (1 + sigma * |x - y|) * exp{-sigma * |x-y|}
#
# where the norm is
#
# |u|_H^2 = int_R sigma^4 f(u)^2 + 2*sigma^2 f'(u)^2 + f''(u)^2 du

mutable struct SteinMaternTensorizedKernel <: SteinTensorizedKernel
    sigma::Float64
end

# default is use sigma value of 1
SteinMaternTensorizedKernel() = SteinMaternTensorizedKernel(1.0)

function getc(ker::SteinMaternTensorizedKernel)
    1 / (8.0 * ker.sigma^3)
end

function ki(ker::SteinMaternTensorizedKernel, x::Float64, y::Float64)
    c = getc(ker)
    d = abs(x-y)
    sigma = ker.sigma
    c * (1 + sigma * d) * exp(-sigma * d)
end

function gradxki(ker::SteinMaternTensorizedKernel, x::Float64, y::Float64)
    c = getc(ker)
    d = abs(x-y)
    sigma = ker.sigma

    c * sigma * (y - x) * exp(-sigma * d)
end

function gradxyki(ker::SteinMaternTensorizedKernel, x::Float64, y::Float64)
    c = getc(ker)
    d = abs(x-y)
    sigma = ker.sigma

    c * sigma * (1 - sigma * d) * exp(-sigma * d)
end
