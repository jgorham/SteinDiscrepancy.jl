# The SteinPolyharmonicSplineKernel for R^d is given by
#
# k(r) = Gamma(d/2 - m) / (2^{2m} pi^{d/2} (m-1)!) r^{2m-d}
#
# if d is odd and
#
# k(r) = (-1)^{m + d/2 - 1} / (2^{2m-1} pi^{d/2} (m-1)!(m-d/2)!) r^{2m-d} log r
#
# if d is even, where r = ||x-y||_2. This is a norm
# for the order m Beppo Levi Space on R^d.
#
# See Section 5.3 of http://arxiv.org/pdf/1204.6448.pdf.

mutable struct SteinPolyharmonicSplineKernel <: SteinKernel
    # the order of the derivative operator
    m::Int
end

# default will be 2
SteinPolyharmonicSplineKernel() = SteinPolyharmonicSplineKernel(2)

function getcons(m::Int, d::Int)
    if (d % 2 == 0)
        (-1)^(m + d/2 - 1) / (2^(2*m - 1) * pi^(d/2) *
            factorial(m - 1) * factorial(int(m - d/2)))
    else
        gamma(d/2 - m) / (2^(2*m) * pi^(d/2) * factorial(m - 1))
    end
end

function k(ker::SteinPolyharmonicSplineKernel, x::Array{Float64,1}, y::Array{Float64,1})
    m = ker.m; d = length(x)
    @assert m > (d/2)
    r = norm(x - y)
    if r == 0.0
        0.0
    end
    c = getcons(m, d)
    if (d % 2 == 0)
        c * r^(2*m - d) * log(r)
    else
        c * r^(2*m - d)
    end
end

function gradxk(ker::SteinPolyharmonicSplineKernel, x::Array{Float64,1}, y::Array{Float64,1})
    m = ker.m; d = length(x)
    @assert m > (d/2)
    r = norm(x - y)
    if r == 0.0
        0.0
    end
    c = getcons(m, d)
    l = 2*m - d
    if (d % 2 == 0)
        c * (x - y) * r^(l - 2) * (l * log(r) + 1.0)
    else
        c * l * (x - y) * r^(l - 2)
    end
end

function gradxyk(ker::SteinPolyharmonicSplineKernel, x::Array{Float64,1}, y::Array{Float64,1})
    m = ker.m; d = length(x)
    @assert m > (d/2)
    r = norm(x - y)

    c = getcons(m, d)
    l = 2*m - d
    if (d % 2 == 0)
        -c * r^(l - 2) * ((l^2 - 2*l + l*d)*log(r) + 2*l + d - 2)
    else
        -c * l * (2*m - 2) * r^(l - 2)
    end
end
