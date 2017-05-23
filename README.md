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

## So how do I use it?

This software has been tested on Julia v0.5. This release implements two
classes of Stein discrepancies: graph Stein discrepancies and kernel Stein
discrepancies.

### Graph Stein discrepancies

Computing the graph Stein discrepancy requires solving a linear program
(LP), and thus you'll need some kind of LP solver installed to use this
software. We use JuMP ([Julia for Mathematical
Programming](https://jump.readthedocs.org/en/latest/)) to interface with
these solvers; any of the supported JuMP LP solvers with do just fine.

Once you have an LP solver installed, computing our measure is easy.
Here's a quick example that will compute the Langevin graph Stein
discrepancy for a bivariate uniform sample:

```
# do the necessary imports
using SteinDiscrepancy: stein_discrepancy
# define the grad log density of target
function gradlogp(x::Array{Float64,1})
    zeros(size(x))
end
# generates 100 points
X = rand(100,2)
# can be a string or a JuMP solver
solver = "clp"
result = stein_discrepancy(points=X, gradlogdensity=gradlogp, solver=solver, method="graph",
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

```
# do the necessary imports
using SteinDiscrepancy: SteinInverseMultiquadricKernel, stein_discrepancy
# define the grad log density of standard normal target
function gradlogp(x::Array{Float64,1})
    -x
end
# grab sample
X = randn(500, 3)
# create the kernel instance
kernel = SteinInverseMultiquadricKernel()
# compute the KSD2
result = stein_discrepancy(points=X, gradlogdensity=gradlogp, method="kernel", kernel=kernel)
# get the final ksd
ksd = sqrt(result.discrepancy2)
```

## Summary of the Code

All code is available in the src directory of the repo. Many examples for
computing the stein_discrepancy are in the test directory.

### Contents of src

* discrepancy - Code for computing Stein discrepancy
* kernels - Commonly used kernels

### Compiling Code in discrepancy/spanner directory

Our C++ code should be compiled when the package is built. However,
if this doesn't work for some reason, you can issue the following
commands to compile the code in discrepancy/spanner:

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