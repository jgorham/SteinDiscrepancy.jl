using BinDeps

@BinDeps.setup

libsteinspanner = library_dependency("libsteinspanner",
                                     aliases=["libstein_spanner.so", "libstein_spanner.1.so", "libstein_spanner.1.0.so", "libstein_spanner.1.0.0.so"])

provides(SimpleBuild,
   (@build_steps begin
        ChangeDirectory(joinpath(dirname(@__FILE__), "../src/discrepancy/spanner"))
        MakeTargets(["libstein_spanner.so"])
    end), libsteinspanner, os = :Unix)

@BinDeps.install Dict(:libsteinspanner => :libsteinspanner)
