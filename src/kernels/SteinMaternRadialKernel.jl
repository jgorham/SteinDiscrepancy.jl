# The SteinMaternRadialKernel is given by
#
# k(x, y) = (1 + gamma * r) * exp{-gamma * r}
#
# where r = ||x-y||. This is a norm for the order
# 2 Sobolev space H_2. This is the Matern covariance
# associated with v = 3/2.
#
# See http://www.gaussianprocess.org/gpml/chapters/RW4.pdf
# or http://arxiv.org/pdf/1204.6448.pdf for more details.

mutable struct SteinMaternRadialKernel <: SteinMaternWeightedKernel
    gamma::Float64
end

# default is use gamma sqrt{3}
SteinMaternRadialKernel() = SteinMaternRadialKernel(sqrt(3.0))

function Q(ker::SteinMaternRadialKernel, x::Array{Float64,1}, y::Array{Float64,1})
    1.0
end

function gradxQ(ker::SteinMaternRadialKernel, x::Array{Float64,1}, y::Array{Float64,1})
    d = length(x)
    zeros(d)
end

function gradxyQ(ker::SteinMaternRadialKernel, x::Array{Float64,1}, y::Array{Float64,1})
    0.0
end
