# The SteinInverseMultiquadricKernel for R^d is given by
#
# k(r) = (c^2 + r^2)^{-beta}
#
# where r = |x-y|_2.

type SteinInverseMultiquadricKernel <: SteinKernel
    beta::Float64       # -beta is exponent
    c2::Float64         # equal to c^2
    SteinInverseMultiquadricKernel(beta, c2) = (
        @assert (0 < c2);
        @assert (0 < beta);
        new(beta, c2)
    )
end

# default will be 1.0 for c^2
SteinInverseMultiquadricKernel(beta::Float64) = SteinInverseMultiquadricKernel(beta, 1.0)
# default will be 0.5 for beta
SteinInverseMultiquadricKernel() = SteinInverseMultiquadricKernel(0.5)

function k(ker::SteinInverseMultiquadricKernel, x::Array{Float64,1}, y::Array{Float64,1})
    r = norm(x - y)
    c2 = ker.c2
    beta = ker.beta
    (c2 + r^2)^(-beta)
end

function gradxk(ker::SteinInverseMultiquadricKernel, x::Array{Float64,1}, y::Array{Float64,1})
    r = norm(x - y)
    c2 = ker.c2
    beta = ker.beta
    -2.0 * beta * (x - y) * (c2 + r^2)^(-beta-1.0)
end

function gradxyk(ker::SteinInverseMultiquadricKernel, x::Array{Float64,1}, y::Array{Float64,1})
    r = norm(x - y)
    c2 = ker.c2
    beta = ker.beta
    d = length(x)
    2 * d * beta * (c2 + r^2)^(-beta-1.0) - 4 * beta * (beta + 1) * r^2 * (c2 + r^2)^(-beta-2.0)
end

# This is reimplemented here, optimized for speed!
function k0(ker::SteinInverseMultiquadricKernel,
            x::Array{Float64, 1},
            y::Array{Float64, 1},
            gradlogpx::Array{Float64, 1},
            gradlogpy::Array{Float64, 1})
    # compute all the core values only once and store them
    c2 = ker.c2
    beta = ker.beta
    d = length(x)

    z = x - y
    r2 = sum(abs2, z)
    base = c2 + r2
    base_beta = base^(-beta)
    base_beta1 = base_beta / base

    coeffk = dot(gradlogpx, gradlogpy)
    coeffgrad = -2.0 * beta * base_beta1

    kterm = coeffk * base_beta
    gradandgradgradterms = coeffgrad * (
        (dot(gradlogpy, z) - dot(gradlogpx, z)) +
        (-d + 2 * (beta + 1) * r2 / base)
    )
    kterm + gradandgradgradterms
end
