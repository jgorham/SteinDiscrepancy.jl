module SteinDiscrepancy

    # We need to make sure the depsjl file is loaded so spanner .so is reachable
    depsjl = joinpath(dirname(@__FILE__), "..", "deps", "deps.jl")
    isfile(depsjl) ? include(depsjl) : error("SteinDiscrepancy not properly ",
        "installed. Please run\nPkg.build(\"SteinDiscrepancy\")")

    using MathProgBase.SolverInterface: AbstractMathProgSolver
    using JuMP

    import Base.rand
    import DataStructures
    import Distributions.TDist
    import Distributions.quantile

    export SteinDiscrete,
           # Utility function for getting the solver
           getsolver,
           # Main method for computing Stein discrepancy
           stein_discrepancy,
           # Result objects returned by affine stein discrepancy methods
           AffineDiscrepancyResult,
           # Result objects returned by langevin_kernel_discrepancy w/o checkpoints
           LangevinKernelResult,
           # Result objects returned by langevin_kernel_discrepancy w/ checkpoints
           LangevinKernelCheckpointsResult,
           # Function for computing spanner edges
           getspanneregdes,
           # Function computing univariate Wasserstein distancen
           wasserstein1d,
           # Function computing Wasserstein distance for discrete distributions
           wassersteindiscrete,
           # Function approximating Wasserstein distance for multivariate Wasserstein
           approxwasserstein,

           ### kernel exports
           SteinKernel,
           SteinParzenARKernel,
           SteinChampionLenardMillsKernel,
           SteinGaussianWeightedKernel,
           SteinGaussianRectangularDomainKernel,
           SteinGaussianPowerKernel,
           SteinGaussianUnboundedPowerKernel,
           SteinGaussianKernel,
           SteinMaternTensorizedKernel,
           SteinMaternRadialKernel,
           SteinPolyharmonicSplineKernel,
           SteinInverseMultiquadricKernel,
           ### kernel utility functions
           k,
           gradxk,
           gradyk,
           gradxyk,
           k0

    ### distributions
    include("SteinDiscrete.jl")

    ### kernels
    include("kernels/SteinKernel.jl")
    include("kernels/SteinTensorizedKernel.jl")
    include("kernels/SteinGaussianWeightedKernel.jl")
    include("kernels/SteinGaussianRectangularDomainKernel.jl")
    include("kernels/SteinGaussianPowerKernel.jl")
    include("kernels/SteinGaussianUnboundedPowerKernel.jl")
    include("kernels/SteinGaussianKernel.jl")
    include("kernels/SteinParzenARKernel.jl")
    include("kernels/SteinChampionLenardMillsKernel.jl")
    include("kernels/SteinPolyharmonicSplineKernel.jl")
    include("kernels/SteinMaternTensorizedKernel.jl")
    include("kernels/SteinMaternWeightedKernel.jl")
    include("kernels/SteinMaternRadialKernel.jl")
    include("kernels/SteinMaternPowerKernel.jl")
    include("kernels/SteinMaternUnboundedPowerKernel.jl")
    include("kernels/SteinInverseMultiquadricKernel.jl")

    ### discrepancy
    include("discrepancy/getsolver.jl")
    # Spanner files and utilities
    include("discrepancy/spanner.jl")
    include("discrepancy/l1spanner.jl")
    include("discrepancy/mincostflow.jl")
    # graph related discrepancies
    include("discrepancy/AffineDiscrepancyResult.jl")
    include("discrepancy/affine_graph_discrepancy.jl")
    include("discrepancy/langevin_graph_discrepancy.jl")
    include("discrepancy/riemannian_langevin_graph_discrepancy.jl")
    # classical related discrepancies
    include("discrepancy/affine_classical_discrepancy.jl")
    include("discrepancy/langevin_classical_discrepancy.jl")
    # kernel related discrepancies
    include("discrepancy/LangevinKernelBaseResult.jl")
    include("discrepancy/langevin_kernel_discrepancy.jl")
    # utilities / addons
    include("discrepancy/wasserstein.jl")
    # the main ONE
    include("discrepancy/stein_discrepancy.jl")
end
