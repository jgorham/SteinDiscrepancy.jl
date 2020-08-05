# The SteinChampionLenardMillsKernel is based on the reference
# "An introduction to abstract splines" written in 1996 by the
# aforementioned authors. The kernel is a kernel for the Sobolev
# space H^2([0,1]) subject to the extra condition that f(0) = f(1) = 0.
# The kernel is of the form
#
# k(x, y) = (1/6) * [(x - y)_+^3 - x (1-y) (x^2 - 2y + y^2)]
#
# and has inner product
#
# <f, g> = \int_[0,1] f''(t) g''(t) dt
#
# In order to extend this arbitrary intervals, we dilate and translate
# the above kernel so that for the interval [0,a], the kernel
# becomes
#
# k_a(x, y) = k(x/a, y/a)
#
# and the inner product is normalized to become
#
# <f,g>_a = (1/a^3) * int_[0,a] f''(t) g''(t) dt
#
# Translating above is trivial.

# utility for grabbing the support bounds
mutable struct SupportBound
    lb::Float64
    ub::Float64
end

mutable struct SteinChampionLenardMillsKernel <: SteinTensorizedKernel
    support::Array{SupportBound,1}
end

# coerce arrays support bounds
SteinChampionLenardMillsKernel(supp::Array{Array{Float64,1},1}) =
    SteinChampionLenardMillsKernel([SupportBound(s[1], s[2]) for s in supp])

# Make it easy to make d dimensional supports
SteinChampionLenardMillsKernel(supp::Array{Float64,1}, d::Int) =
    SteinChampionLenardMillsKernel([supp for i in 1:d])

# Make it easy to make 1-d supports
SteinChampionLenardMillsKernel(supp::Array{Float64,1}) =
    SteinChampionLenardMillsKernel(supp, 1)

# Assumes two arrays are [lb] and [ub]
SteinChampionLenardMillsKernel(lb::Array{Float64,1}, ub::Array{Float64,1}) =
    SteinChampionLenardMillsKernel([SupportBound(lb[i], ub[i]) for i=1:length(lb)])

# default is just supprt [0,1]
SteinChampionLenardMillsKernel() = SteinChampionLenardMillsKernel([[0.0,1.0]])

#### utility methods
# returns the parameters to translate and dilate the kernel to appropriate interval
function getaffinetransform(ker::SteinChampionLenardMillsKernel, i::Int)
    bt = ker.support[i]
    lb = bt.lb; ub = bt.ub
    lb, (ub - lb)
end

# this is kernel k with domain [0,1]
function k01(x::Float64, y::Float64)
    (max(x - y, 0)^3 - x * (1-y) * (x^2 - 2*y + y^2)) / 6
end

function gradxk01(x::Float64, y::Float64)
    (3 * (x-y)^2 * Int(x >= y) - (1-y) * (3*x^2 - 2*y + y^2)) / 6
end

function gradxyk01(x::Float64, y::Float64)
    0.5 * x^2 + 0.5 * y^2 - max(x,y) + (1/3)
end

### Mandatory methods to implement
function ki(ker::SteinChampionLenardMillsKernel, x::Float64, y::Float64, i::Int)
    c, a = getaffinetransform(ker, i)
    k01((x - c) / a, (y - c) / a)
end

function gradxki(ker::SteinChampionLenardMillsKernel, x::Float64, y::Float64, i::Int)
    c, a = getaffinetransform(ker, i)
    (1/a) * gradxk01((x - c) / a, (y - c) / a)
end

function gradxyki(ker::SteinChampionLenardMillsKernel, x::Float64, y::Float64, i::Int)
    c, a = getaffinetransform(ker, i)
    (1/a^2) * gradxyk01((x - c) / a, (y - c) / a)
end
