# Returns solver object associated with string name
#
# Args:
#  solver - Optimization problem solver name in
#   {"gurobi","gurobi_no_crossover","clp","glpk","ecos","scs"}
function getsolver(solver::AbstractString)
    solver = lowercase(solver)
    if solver == "gurobi"
        # Gurobi Options
        # BarConvTol: tolerance in relative primal to dual error for barrier method;
        #  default: 1e-8
        # Crossover: how to convert barrier solution into basic solution
        #  -1 (default): automatic setting; 0: disable crossover
        # Method: solution method
        #  -1 (default): automatic setting; 2: barrier method
        # NumericFocus: degree of numerical issue detection and management
        #  0 (default): automatic setting;
        #  Settings 1-3 imply increasing care in numerical computations
        # Presolve: controls the presolve level
        #  -1 (default): automatic setting
        #  0: disabled; 1: conservative; 2: aggressive
        eval(Expr(:import,:Gurobi))
        solver_object = Main.Gurobi.GurobiSolver()
    elseif solver == "gurobi_no_crossover"
        eval(Expr(:import,:Gurobi))
        solver_object = Main.Gurobi.GurobiSolver(Method=2,Crossover=0)
    elseif solver == "gurobi_one_thread"
        eval(Expr(:import,:Gurobi))
        solver_object = Main.Gurobi.GurobiSolver(Threads=1)
    elseif solver == "clp"
        eval(Expr(:import,:Clp))
        # LogLevel: set to 1, 2, 3, or 4 for increasing output (default 0)
        # PresolveType: set to 1 to disable presolve
        # SolveType: choose the solution method:
        #  0 - dual simplex, 1 - primal simplex,
        #  3 - barrier with crossover to optimal basis,
        #  4 - barrier without crossover to optimal basis
        #  5 - automatic
        solver_object = Main.Clp.ClpSolver(LogLevel=4)
    elseif solver == "glpk"
        eval(Expr(:import,:GLPKMathProgInterface))
        # GLPK Options
        # msg_level: verbosity level in {0,...,4}; defaults to 0 (no output)
        # presolve: presolve LP?; defaults to false
        solver_object = Main.GLPKMathProgInterface.GLPKSolverLP()
    elseif solver == "scs"
        eval(Expr(:import,:SCS))
        solver_object = Main.SCS.SCSSolver()
    elseif solver == "ecos"
        eval(Expr(:import,:ECOS))
        solver_object = Main.ECOS.ECOSSolver()
    else
        error("Unknown solver $solver_object requested.")
    end
end
