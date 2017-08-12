# The returned result from a langevin kernel discrepancy

abstract type LangevinKernelBaseResult end

type LangevinKernelResult <: LangevinKernelBaseResult
    # n x p matrix of distinct sample points
    points::Array{Float64}
    # n x 1 vector of weights associated with each sample
    weights::Array{Float64}
    # the sum of the kernels (squared) that bound the IPM
    discrepancy2::Float64
    # Time taken to set up and solve each optimization program
    solvetime::Float64
end

type LangevinKernelCheckpointsResult <: LangevinKernelBaseResult
    # n x p matrix of distinct sample points
    points::Array{Float64}
    # n x 1 vector of weights associated with each sample
    weights::Array{Float64}
    # the sum of the kernels (squared) that bound the IPM by checkpoints
    discrepancy2::Array{Float64}
    #checkpoints passed into the function
    checkpoints::Array{Int}
    # Time taken to set up and solve each optimization program
    solvetime::Float64
end
