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

# Graph Stein discrepancy unbounded test
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
