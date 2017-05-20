# SteinKernel

abstract SteinKernel

## METHODS TO IMPLEMENT ##
function k(kernel::SteinKernel, x::Array{Float64, 1}, y::Array{Float64, 1})
    error("Must implement the method k")
end

function gradxk(kernel::SteinKernel, x::Array{Float64, 1}, y::Array{Float64, 1})
    error("Must implement the method gradxk")
end

function gradxyk(kernel::SteinKernel, x::Array{Float64, 1}, y::Array{Float64, 1})
    error("Must implement the method gradxyk")
end

# Computes \grad_y k(x, y) = \grad_x k(y, x)
function gradyk(kernel::SteinKernel, x::Array{Float64, 1}, y::Array{Float64, 1})
    gradxk(kernel, y, x)
end

# This is the Chris Oates Langevin Stein kernel
function k0(kernel::SteinKernel,
            x::Array{Float64, 1},
            y::Array{Float64, 1},
            gradlogpx::Array{Float64, 1},
            gradlogpy::Array{Float64, 1})
    k(kernel, x, y) * dot(gradlogpx, gradlogpy) +
        dot(gradlogpx, gradyk(kernel, x, y)) +
        dot(gradlogpy, gradxk(kernel, x, y)) +
        gradxyk(kernel, x, y)
end
