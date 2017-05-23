using Base.Test
using Clp
using SteinDiscrepancy:
    SteinInverseMultiquadricKernel,
    stein_discrepancy,
    wassersteindiscrete

# import data
include("data/test_data.jl")

function uniform_gradlogp(x::Array{Float64,1})
    zeros(size(x))
end

function gaussian_gradlogp(x::Array{Float64,1})
    -x
end
# define the covariance terms for the Riemannian-Langevin diffusion
function volatility_covariance(x::Array{Float64,1})
    p = length(x)
    sqrt(1 + sum(x.^2)) * eye(p)
end
function grad_volatility_covariance(x::Array{Float64,1})
    x / sqrt(1 + sum(x.^2))
end

imqkernel = SteinInverseMultiquadricKernel()

# Graph Stein discrepancy bounded test
@testset "Univariate Graph discrepancy test" begin
    res = stein_discrepancy(points=UNIFORM_TESTDATA[:,1],
                            gradlogdensity=uniform_gradlogp,
                            solver="clp",
                            method="graph",
                            supportlowerbounds=[0.0],
                            supportupperbounds=[1.0])

    @test_approx_eq_eps res.objectivevalue[1] 0.15 1e-5
end
@testset "Multivariate Graph discrepancy test" begin
    res = stein_discrepancy(points=UNIFORM_TESTDATA,
                            gradlogdensity=uniform_gradlogp,
                            solver="clp",
                            method="graph",
                            supportlowerbounds=zeros(size(UNIFORM_TESTDATA,2)),
                            supportupperbounds=ones(size(UNIFORM_TESTDATA,2)))

    @test_approx_eq_eps res.objectivevalue[1] 0.308 1e-5
    @test_approx_eq_eps res.objectivevalue[2] 0.31 1e-5
end
@testset "Riemannian Graph discrepancy test" begin
    res = stein_discrepancy(points=GAUSSIAN_TESTDATA,
                            gradlogdensity=gaussian_gradlogp,
                            operator="riemannian-langevin",
                            solver="clp",
                            method="graph",
                            volatility_covariance=volatility_covariance,
                            grad_volatility_covariance=grad_volatility_covariance)

    @test_approx_eq_eps res.objectivevalue[1] 6.19106 1e-5
    @test_approx_eq_eps res.objectivevalue[2] 7.05982 1e-5
end

# Kernel Stein discrepancy test
@testset "Univariate Kernel discrepancy test" begin
    res = stein_discrepancy(points=GAUSSIAN_TESTDATA[:,1],
                            gradlogdensity=gaussian_gradlogp,
                            method="kernel",
                            kernel=imqkernel)

    @test_approx_eq_eps res.discrepancy2 0.269298 1e-5
end
@testset "Multivariate Kernel discrepancy test" begin
    res = stein_discrepancy(points=GAUSSIAN_TESTDATA,
                            gradlogdensity=gaussian_gradlogp,
                            method="kernel",
                            kernel=imqkernel)

    @test_approx_eq_eps res.discrepancy2 0.807566 1e-5
end

# Wassersteindiscrete test
@testset "Wasserstein bivariate discrete test" begin
    (emd, numnodes, numedges, status) =
        wassersteindiscrete(xpoints=GAUSSIAN_TESTDATA,
                            ypoints=UNIFORM_TESTDATA,
                            solver="clp")

    @test_approx_eq_eps emd 1.56 1e-5
    @test status == :Optimal
end
