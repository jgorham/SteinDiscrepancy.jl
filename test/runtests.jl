using Test
using Clp
using LinearAlgebra
using SteinDiscrepancy:
    SteinInverseMultiquadricKernel,
    SteinRectangularDomainMetaKernel,
    stein_discrepancy,
    ksd,
    gsd,
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
    sqrt(1 + sum(x.^2)) * I
end
function grad_volatility_covariance(x::Array{Float64,1})
    x / sqrt(1 + sum(x.^2))
end

imqkernel = SteinInverseMultiquadricKernel()
imqrectkernel = SteinRectangularDomainMetaKernel(imqkernel, [0., 0.], [1., 1.])
solver = Clp.ClpSolver(LogLevel=4)

# Graph Stein discrepancy bounded test
@testset "Univariate Graph discrepancy test" begin
    res = gsd(points=UNIFORM_TESTDATA[:,1],
              gradlogdensity=uniform_gradlogp,
              solver=solver,
              supportlowerbounds=[0.0],
              supportupperbounds=[1.0])

    @test res.objectivevalue[1] ≈ 0.15 atol=1.0e-5
end
@testset "Multivariate Graph discrepancy test" begin
    res = stein_discrepancy(points=UNIFORM_TESTDATA,
                            gradlogdensity=uniform_gradlogp,
                            solver=solver,
                            method="graph",
                            supportlowerbounds=zeros(size(UNIFORM_TESTDATA,2)),
                            supportupperbounds=ones(size(UNIFORM_TESTDATA,2)))

    @test res.objectivevalue[1] ≈ 0.308 atol=1.0e-5
    @test res.objectivevalue[2] ≈ 0.31 atol=1.0e-5
end
@testset "Riemannian Graph discrepancy test" begin
    res = stein_discrepancy(points=GAUSSIAN_TESTDATA,
                            gradlogdensity=gaussian_gradlogp,
                            operator="riemannian-langevin",
                            solver=solver,
                            method="graph",
                            volatility_covariance=volatility_covariance,
                            grad_volatility_covariance=grad_volatility_covariance)

    @test res.objectivevalue[1] ≈ 6.19106 atol=1.0e-5
    @test res.objectivevalue[2] ≈ 7.05982 atol=1.0e-5
end

# Kernel Stein discrepancy test
@testset "Univariate Kernel discrepancy test" begin
    res = stein_discrepancy(points=GAUSSIAN_TESTDATA[:,1],
                            gradlogdensity=gaussian_gradlogp,
                            method="kernel",
                            kernel=imqkernel)

    @test res.discrepancy2 ≈ 0.269298 atol=1.0e-5
end
@testset "Multivariate Kernel discrepancy test" begin
    res = ksd(points=GAUSSIAN_TESTDATA,
              gradlogdensity=gaussian_gradlogp,
              kernel=imqkernel)

    @test res.discrepancy2 ≈ 0.807566 atol=1.0e-5
end

# Wassersteindiscrete test
@testset "Wasserstein bivariate discrete test" begin
    (emd, numnodes, numedges, status) =
        wassersteindiscrete(xpoints=GAUSSIAN_TESTDATA,
                            ypoints=UNIFORM_TESTDATA,
                            solver=solver)

    @test emd ≈ 1.56 atol=1e-5
    @test status == :Optimal
end

# Stein Meta Kernel test
@testset "Rectangular domain meta kernel test" begin
    res = ksd(points=UNIFORM_TESTDATA,
              gradlogdensity=uniform_gradlogp,
              kernel=imqrectkernel)

    @test res.discrepancy2 ≈ 0.0064761153 atol=1.0e-8
end
