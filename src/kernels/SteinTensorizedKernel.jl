# SteinTensorizedKernel
#
# These represent a family of kernels of the form
#
# K(x, y) = \prod_i ki(x_i, y_i)
#
# where ki(x,y) is a kernel in one dimension. In general, ki
# will be the same for each dimension i; the exception being
# when the support is not the same amongst all dimensions.

abstract type SteinTensorizedKernel <: SteinKernel end

### METHODS TO IMPLEMENT
function ki(ker::SteinTensorizedKernel, x::Float64, y::Float64)
    error("Must implement the method ki")
end

function gradxki(ker::SteinTensorizedKernel, x::Float64, y::Float64)
    error("Must implement the method gradxki")
end

function gradxyki(ker::SteinTensorizedKernel, x::Float64, y::Float64)
    error("Must implement the method gradxyki")
end

### INDUCED METHODS
function ki(ker::SteinTensorizedKernel, x::Float64, y::Float64, i::Int)
    # if not overriden, we assume each dimension is the same
    ki(ker, x, y)
end

function gradxki(ker::SteinTensorizedKernel, x::Float64, y::Float64, i::Int)
    gradxki(ker, x, y)
end

function gradxyki(ker::SteinTensorizedKernel, x::Float64, y::Float64, i::Int)
    gradxyki(ker, x, y)
end

# this is a helpful utility for computing \prod_{j != i} kj(x_j, y_j)
function k_minusi(ker::SteinTensorizedKernel,
                  x::Array{Float64, 1},
                  y::Array{Float64, 1},
                  i::Int)
    d = length(x)
    noti = filter(x -> x != i, 1:d)
    prod([ki(ker, x[j], y[j], j) for j in noti])
end

function k(ker::SteinTensorizedKernel, x::Array{Float64,1}, y::Array{Float64,1})
    d = length(x)
    prod([ki(ker, x[i], y[i], i) for i=1:d])
end

function gradxk(ker::SteinTensorizedKernel, x::Array{Float64,1}, y::Array{Float64,1})
    d = length(x)
    [gradxki(ker, x[i], y[i], i) * k_minusi(ker, x, y, i) for i=1:d]
end

function gradxyk(ker::SteinTensorizedKernel, x::Array{Float64,1}, y::Array{Float64,1})
    d = length(x)
    sum([gradxyki(ker, x[i], y[i], i) * k_minusi(ker, x, y, i) for i=1:d])
end
