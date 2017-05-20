# The SteinGaussianKernel has base kernel of the form:
#
# k(x, y) = exp(-1/(2*beta)*||x-y||^2)
#
# where ||.|| is the L2 norm.

type SteinGaussianKernel <: SteinGaussianWeightedKernel
    # the beta parameter
    beta::Float64
end

# have default parameter of beta = 1.0
SteinGaussianKernel() = SteinGaussianKernel(1.0)

function Q(ker::SteinGaussianKernel, x::Array{Float64, 1}, y::Array{Float64, 1})
    1.0
end

function dxj_Q(kernel::SteinGaussianKernel, x::Array{Float64,1}, y::Array{Float64,1}, j::Int64)
    0.0
end

function dxjyj_Q(kernel::SteinGaussianKernel, x::Array{Float64,1}, y::Array{Float64,1}, j::Int64)
    0.0
end
