# The SteinParzenARKernel is a kernel derived by Parzen in 1961,
# which arises from a second order autoregressive process. Its
# inner product is for the Sobolev space H^2([a, b]) for any
# a < b.
#
# k(x, y) = (4*alpha*gamma^2)^-1 * exp{-alpha*|x - y|} cos(w*|x-y|)
#   + (alpha / w) * sin(w*|x-y|)
#
# where w^2 = gamma^2 - alpha^2 > 0. The associated norm is
#
# |u|_H = 4*alpha*gamma^2*u(a)^2 + 4*alpha*u'(a)^2 +
#  \int_[a,b] (u''(t) + 2*alpha^2*u'(t) + gamma^2*u(t))^2

type SteinParzenARKernel <: SteinTensorizedKernel
    # the alpha parameter
    alpha::Float64
    # the gamma parameter
    gamma::Float64
end

# have default parameters of alpha = 1/sqrt(2) and gamma = 1.0
SteinParzenARKernel() = SteinParzenARKernel(1.0/sqrt(2), 1.0)

# utility to get w
function getw(ker::SteinParzenARKernel)
    sqrt(ker.gamma^2 - ker.alpha^2)
end

# utility method for wacky constant
function getc1(ker::SteinParzenARKernel)
    1.0 / (4.0 * ker.alpha * ker.gamma^2)
end

function ki(ker::SteinParzenARKernel, x::Float64, y::Float64)
    w = getw(ker)
    c = getc1(ker)
    alpha = ker.alpha

    c * exp(-alpha * abs(x-y)) * cos(w * abs(x-y)) + (alpha/w) * sin(w * abs(x-y))
end

function gradxki(ker::SteinParzenARKernel, x::Float64, y::Float64)
    w = getw(ker)
    c = getc1(ker)
    alpha = ker.alpha
    s = sign(x - y)

    -c * s * exp(-alpha * abs(x-y)) * (
        alpha * cos(w * abs(x-y)) +
        w * sin(w * abs(x-y))
    ) + alpha * s * cos(w * abs(x-y))
end

function gradxyki(ker::SteinParzenARKernel, x::Float64, y::Float64)
    w = getw(ker)
    c = getc1(ker)
    alpha = ker.alpha
    s = sign(x - y)

    -c * exp(-alpha * abs(x-y)) * (
        (alpha^2 - w^2) * cos(w * abs(x-y)) +
        2 * alpha * w * sin(w * abs(x-y))
    ) + alpha * w * sin(w * abs(x-y))
end
