using Base.Test
using Clp
using SteinDiscrepancy:
    SteinInverseMultiquadricKernel,
    stein_discrepancy

# import data
include("data/test_data.jl")

function uniform_gradlogp(x::Array{Float64,1})
    zeros(size(x))
end

function gaussian_gradlogp(x::Array{Float64,1})
    -x
end

imqkernel = SteinInverseMultiquadricKernel()

# Graph Stein discrepancy bounded test
@testset "Graph discrepancy test" begin
    res = stein_discrepancy(points=UNIFORM_TESTDATA,
                            gradlogdensity=uniform_gradlogp,
                            solver="clp",
                            method="graph",
                            supportlowerbounds=zeros(size(UNIFORM_TESTDATA,2)),
                            supportupperbounds=ones(size(UNIFORM_TESTDATA,2)))

    @test_approx_eq_eps res.objectivevalue[1] 0.308 1e-5
    @test_approx_eq_eps res.objectivevalue[2] 0.31 1e-5
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
