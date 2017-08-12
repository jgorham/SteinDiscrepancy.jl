# SteinDiscrepancy.jl

## What's a Stein discrepancy?

To improve the efficiency of Monte Carlo estimation, practitioners are
turning to biased Markov chain Monte Carlo procedures that trade off
asymptotic exactness for computational speed. The reasoning is sound: a
reduction in variance due to more rapid sampling can outweigh the bias
introduced. However, the inexactness creates new challenges for sampler and
parameter selection, since standard measures of sample quality like
effective sample size do not account for asymptotic bias. To address these
challenges, we introduced new computable quality measures, based on Stein's
method, that quantify the maximum discrepancy between sample and target
expectations over large classes of test functions. We call these measures
Stein discrepancies.

For a more detailed explanation, take a peek at the latest papers:

[Measuring Sample Quality with Diffusions](https://arxiv.org/abs/1611.06972),
[Measuring Sample Quality with Kernels](https://arxiv.org/abs/1703.01717).

These build on previous work from

[Measuring Sample Quality with Stein's Method](http://arxiv.org/abs/1506.03039)

and its companion paper

[Multivariate Stein Factors for a Class of Strongly Log-concave
Distributions](http://arxiv.org/abs/1512.07392).

These latter two papers are a more gentle introduction describing how the
Stein discrepancy bounds standard probability metrics like the
[Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric).

Code built on `SteinDiscrepancy.jl` recreating all experiments in the above
papers can be found at the repo
[stein_discrepancy](https://github.com/jgorham/stein_discrepancy).

## Where has it been used?

Since its introduction in [Measuring Sample Quality with Stein's
Method](http://arxiv.org/abs/1506.03039), the Stein discrepancy has been
incorporated into a variety of applications including:

1. Hypothesis testing
   * [A Kernelized Stein Discrepancy for Goodness-of-fit Tests and Model Evaluation](https://arxiv.org/abs/1602.03253)
   * [A Kernel Test of Goodness of Fit](https://arxiv.org/abs/1602.02964)
2. Variational inference
   * [Operator Variational Inference](https://arxiv.org/abs/1610.09033)
   * [Two Methods For Wild Variational Inference](https://arxiv.org/abs/1612.00081)
   * [Approximate Inference with Amortised MCMC](https://arxiv.org/abs/1702.08343)
   * [Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm](https://arxiv.org/abs/1608.04471)
3. Importance sampling
   * [Black-box Importance Sampling](https://arxiv.org/abs/1610.05247)
   * [Stein Variational Adaptive Importance Sampling](https://arxiv.org/abs/1704.05201)
4. Training generative adversarial networks (GANs)
   * [Learning to Draw Samples: With Application to Amortized MLE for Generative Adversarial Learning](https://arxiv.org/abs/1611.01722)
5. Training variational autoencoders (VAEs)
   * [Stein Variational Autoencoder](https://arxiv.org/abs/1704.05155)
6. Sample quality measurement
   * [Measuring Sample Quality with Stein's Method](http://arxiv.org/abs/1506.03039)
   * [Measuring Sample Quality with Diffusions](https://arxiv.org/abs/1611.06972)
   * [Measuring Sample Quality with Kernels](https://arxiv.org/abs/1703.01717)

## So how do I use it?

This software has been tested on Julia v0.6. This release implements two
classes of Stein discrepancies: graph Stein discrepancies and kernel Stein
discrepancies.

### Graph Stein discrepancies

Computing the graph Stein discrepancy requires solving a linear program
(LP), and thus you'll need some kind of LP solver installed to use this
software. We use JuMP ([Julia for Mathematical
Programming](https://jump.readthedocs.org/en/latest/)) to interface with
these solvers; any of the supported JuMP LP solvers with do just fine.

Once you have an LP solver installed, computing our measure is easy.
Below we'll first show how to compute the Langevin graph Stein discrepancy
for a univariate Gaussian target:

```julia
# import the gsd function (graph Stein discrepancy)
using SteinDiscrepancy: gsd
# define the grad log density of univariate Gaussian (we always expect vector inputs!)
function gradlogp(x::Array{Float64,1})
    -x
end
# generates 100 points from N(0,1)
X = randn(100)
# can be a string or a JuMP solver
solver = "clp"
result = gsd(points=X, gradlogdensity=gradlogp, solver=solver)
graph_stein_discrepancy = result.objectivevalue[1]
```

Here's another example that will compute the Langevin graph Stein
discrepancy for a bivariate uniform sample (notice this target distribution
has a bounded support so these bounds become part of the input parameters):

```julia
# do the necessary imports
using SteinDiscrepancy: gsd
# define the grad log density of bivariate uniform target
function gradlogp(x::Array{Float64,1})
    zeros(size(x))
end
# generates 100 points from Unif([0,1]^2)
X = rand(100,2)
# can be a string or a JuMP solver
solver = "clp"
result = gsd(points=X,
             gradlogdensity=gradlogp,
             solver=solver,
             supportlowerbounds=zeros(2),
             supportupperbounds=ones(2))
discrepancy = vec(result.objectivevalue)
```

The variable `discrepancy` here will encode the Stein discrepancy along each
dimension. The final discrepancy is just the sum of this vector.

### Kernel Stein discrepancies

Computing the kernel Stein discrepancies does not require a LP solver. In
lieu of a solver, you will need to specify a kernel function. Many common
kernels are already implemented in `src/kernels`, but if yours is not there
for some reason, feel free to inherit from the `SteinKernel` type and roll
your own.

With a kernel in hand, computing the kernel Stein discrepancy is easy:

```julia
# import the kernel stein discrepancy function and kernel to use
using SteinDiscrepancy: SteinInverseMultiquadricKernel, ksd
# define the grad log density of standard normal target
function gradlogp(x::Array{Float64,1})
    -x
end
# grab sample
X = randn(500, 3)
# create the kernel instance
kernel = SteinInverseMultiquadricKernel()
# compute the KSD2
result = ksd(points=X, gradlogdensity=gradlogp, kernel=kernel)
# get the final ksd
kernel_stein_discrepancy = sqrt(result.discrepancy2)
```

If your target has constrained support, you should simply use a kernel that 
respects these constraints (no `supportlowerbounds` and `supportupperbounds` 
arguments are needed).  See `SteinGaussianRectangularDomainKernel` in the 
`src/kernels` code directory for an example.

## Summary of the Code

All code is available in the src directory of the repo. Many examples for
computing the stein_discrepancy are in the test directory.

### Contents of src

* discrepancy - Code for computing Stein discrepancy
* kernels - Commonly used kernels

### Compiling Code in discrepancy/spanner directory

**Our C++ code should be compiled automatically when the package is
built**. However, if this doesn't work for some reason, you can issue the
following commands to compile the code in `src/discrepancy/spanner`:

```
cd <PACKAGE_DIR>/src/discrepancy/spanner
make
make clean
```

The last step isn't necessary, but it will remove some superfluous
files. If you want to kill everything made in the build process, just run

```
make distclean
```
