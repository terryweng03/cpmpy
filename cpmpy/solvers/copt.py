#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## copt.py
##

from typing import Optional, List, Callable, Iterable
import warnings
import cpmpy as cp
from cpmpy.transformations.negation import push_down_negation, push_down_negation_objective

from .solver_interface import SolverInterface, SolverStatus, ExitStatus, Callback
from ..exceptions import NotSupportedError
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.globalfunctions import FloatSum
from ..expressions.utils import argvals, argval, is_any_list, is_num, is_int
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl, intvar
from ..expressions.globalconstraints import DirectConstraint
from ..transformations.comparison import only_numexpr_equality
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.get_variables import get_variables
from ..transformations.linearize import linearize_constraint, linearize_reified_variables, only_positive_bv, only_positive_bv_wsum_const, decompose_linear, decompose_linear_objective
from ..transformations.normalize import toplevel_list
from ..transformations.reification import only_implies, reify_rewrite, only_bv_reifies
from ..transformations.safening import no_partial_functions, safen_objective

try:
    import coptpy
    COPT_ENV = None
except ImportError:
    pass


class CPM_copt(SolverInterface):
    """
    Interface to COPT's Python API (coptpy)

    Creates the following attributes (see parent constructor for more):

    - ``copt_model``: object, COPT's model object

    The :class:`~cpmpy.expressions.globalconstraint.DirectConstraint`, when used, calls a function on the ``copt_model`` object.

    """

    # COPT natively supports min/max/abs as general constraints; mul (bilinear) is supported
    # via quadratic constraints. 'pow' is not natively supported by COPT and is decomposed.
    supported_global_constraints = frozenset({"min", "max", "abs", "mul"})
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def supported():
        return CPM_copt.installed()

    @staticmethod
    def installed():
        try:
            import coptpy
            global COPT_ENV
            envconfig = coptpy.EnvrConfig()
            envconfig.set('nobanner', '1')
            COPT_ENV = coptpy.Envr(envconfig)
            return True
        except ModuleNotFoundError:
            return False
        except Exception as e:
            raise e

    @staticmethod
    def version() -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        """
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("coptpy")
        except PackageNotFoundError:
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
            cpm_model: a CPMpy Model()
            subsolver: None, not used
        """
        if not self.installed():
            raise ModuleNotFoundError("CPM_copt: Install the python package 'cpmpy[copt]' to use this solver interface.")

        # TODO: subsolver could be a COPT_ENV if a user would want to hand one over
        self.copt_model = COPT_ENV.createModel("cpmpy")
        try:
            self.copt_model.setParam("Logging", 0)
        except Exception:
            pass
        self._obj_offset = 0
        self.objective_ = None
        self._has_objective = False

        # initialise everything else and post the constraints/objective
        # it is sufficient to implement add() and minimize/maximize() below
        super().__init__(name="copt", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.copt_model


    def solve(self, time_limit:Optional[float]=None, solution_callback:Optional[Callable]=None, display:Optional[Callback]=None, **kwargs):
        """
            Call the COPT solver

            Arguments:
                time_limit (float, optional):  maximum solve time in seconds
                solution_callback:             COPT callback object (subclass of coptpy.CallbackBase),
                                               takes precedence over ``display`` when both are set.
                display:                       generic solution callback for use during optimization.
                                               either a list of CPMpy expressions, OR a callback function which
                                               gets called after the variable-value mapping of the intermediate solution.
                                               default/None: nothing is displayed
                **kwargs:                      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            Examples of COPT supported arguments include (case-sensitive names matching ``COPT.Param.*``):

            - ``Threads`` : int
            - ``MipTasks`` : int
            - ``HeurLevel`` : int
            - ``CutLevel`` : int
            - ``Presolve`` : int

            For a full list of COPT parameters, please visit https://guide.copt.de
        """
        from coptpy import COPT

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # edge case, empty model, ensure the solver has something to solve
        if not len(self.user_vars):
            self.add(intvar(1, 1) == 1)

        # set time limit
        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            self.copt_model.setParam("TimeLimit", time_limit)

        # handle solution callbacks
        callback = None
        if solution_callback is not None:
            callback = solution_callback
        elif display is not None:
            callback = self._get_callback(display, events=[COPT.CBCONTEXT_MIPSOL])

        # call the solver, with parameters
        for param, val in kwargs.items():
            self.copt_model.setParam(param, val)

        # call COPT, optionally with callback
        if callback is not None:
            self.copt_model.setCallback(callback, COPT.CBCONTEXT_MIPSOL)
        self.copt_model.solve()
        copt_status = self.copt_model.status

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.copt_model.solvingtime

        # translate exit status
        if copt_status == COPT.OPTIMAL:
            # COP
            if self.has_objective():
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            # CSP
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif copt_status == COPT.INFEASIBLE:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif copt_status == COPT.UNBOUNDED or copt_status == COPT.INF_OR_UNB:
            # unbounded models are not really representable in CPMpy's CSP/COP view;
            # treat as unknown to avoid claiming unsat
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        elif copt_status in (COPT.TIMEOUT, COPT.NODELIMIT, COPT.ITERLIMIT, COPT.INTERRUPTED):
            if self.copt_model.hasSol:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            else:
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        elif copt_status == COPT.UNSTARTED:
            self.cpm_status.exitstatus = ExitStatus.NOT_RUN
        elif copt_status in (COPT.NUMERICAL, COPT.IMPRECISE, COPT.UNFINISHED):
            self.cpm_status.exitstatus = ExitStatus.ERROR
        else:  # another?
            raise NotImplementedError(
                f"Translation of copt status {copt_status} to CPMpy status not implemented")  # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                solver_val = self.solver_var(cpm_var).x
                if cpm_var.is_bool():
                    cpm_var._value = solver_val >= 0.5
                else:
                    cpm_var._value = round(solver_val)
            if self.has_objective():
                assert self.objective_ is not None
                val = self.objective_.value()
                if val is not None and round(val) == val:
                    self.objective_value_ = int(val)
                else:  # FloatSum, float value must be read through FloatSum.value()
                    self.objective_value_ = None

        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var._value = None

        return has_sol


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
            or returns a constant if the variable is a constant
        """
        if isinstance(cpm_var, _NumVarImpl):
            name = cpm_var.name
            revar = self._varmap.get(name)
            if revar is not None:
                return revar

            # not yet created, make a new solver var
            from coptpy import COPT
            if cpm_var.is_bool():
                if isinstance(cpm_var, NegBoolView):
                    raise NotSupportedError("Negative literals should not be left as part of any equation. Please report.")
                revar = self.copt_model.addVar(lb=0, ub=1, vtype=COPT.BINARY, name=name)
            else:
                revar = self.copt_model.addVar(lb=cpm_var.lb, ub=cpm_var.ub, vtype=COPT.INTEGER, name=str(cpm_var))
            self._varmap[name] = revar
            return revar

        if is_int(cpm_var):  # shortcut, eases posting constraints
            return cpm_var

        raise NotImplementedError("Not a known var {}".format(cpm_var))


    def minimize(self, expr: Expression | FloatSum) -> None:
        self.objective(expr, minimize=True)

    def maximize(self, expr: Expression | FloatSum) -> None:
        self.objective(expr, minimize=False)

    def objective(self, expr: Expression | FloatSum, minimize: bool = True) -> None:
        """
            Post the given expression to the solver as objective to minimize/maximize

            'objective()' can be called multiple times, only the last one is stored

            .. note::
                technical side note: any constraints created during conversion of the objective
                are permanently posted to the solver
        """
        from coptpy import COPT

        self.objective_ = expr
        self._has_objective = True

        if isinstance(expr, FloatSum):
            ws, vs, const = expr.components()
            self.user_vars.update(vs)  # save user variables
            self._obj_offset = const

            import coptpy
            copt_obj = coptpy.quicksum(w * sv for w, sv in zip(ws, self.solver_vars(vs))) + const
        else:
            # save user variables
            get_variables(expr, self.user_vars)

            # transform objective
            obj, safe_cons = safen_objective(expr)
            obj, decomp_cons = decompose_linear_objective(obj,
                                                          supported=self.supported_global_constraints,
                                                          supported_reified=self.supported_reified_global_constraints,
                                                          csemap=self._csemap)
            obj = push_down_negation_objective(obj)
            obj, flat_cons = flatten_objective(obj, csemap=self._csemap)
            obj, self._obj_offset = only_positive_bv_wsum_const(obj)  # remove negboolviews, track offset

            self.add(safe_cons + decomp_cons + flat_cons)

            # make objective function or variable and post
            copt_obj = self._make_numexpr(obj) + self._obj_offset

        if minimize:
            self.copt_model.setObjective(copt_obj, sense=COPT.MINIMIZE)
        else:
            self.copt_model.setObjective(copt_obj, sense=COPT.MAXIMIZE)
        self.copt_model.update()

    def has_objective(self):
        # COPT's fresh models default to a MINIMIZE sense even before an objective is set,
        # so we track objective posting explicitly.
        return self._has_objective

    def _make_numexpr(self, cpm_expr):
        """
            Turns a numeric CPMpy 'flat' expression into a solver-specific
            numeric expression

            Used especially to post an expression as objective function
        """
        import coptpy

        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # sum
        if cpm_expr.name == "sum":
            return coptpy.quicksum(self.solver_vars(cpm_expr.args))
        if cpm_expr.name == "sub":
            a,b = self.solver_vars(cpm_expr.args)
            return a - b
        # wsum
        if cpm_expr.name == "wsum":
            return coptpy.quicksum(w * self.solver_var(var) for w, var in zip(*cpm_expr.args))

        raise NotImplementedError("copt: Not a known supported numexpr {}".format(cpm_expr))

    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the :ref:`Adding a new solver` docs on readthedocs for more information.

            :param cpm_expr: CPMpy expression, or list thereof
            :type cpm_expr: Expression or list of Expression

            :return: list of Expression
        """
        # apply transformations, then post internally
        # expressions have to be linearized to fit in MIP model. See /transformations/linearize
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"mod", "div", "element", "nd_element"})  # linearize and decompose expect safe exprs
        cpm_cons = push_down_negation(cpm_cons)
        cpm_cons = decompose_linear(cpm_cons,
                                    supported=self.supported_global_constraints,
                                    supported_reified=self.supported_reified_global_constraints,
                                    csemap=self._csemap)
        cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']), csemap=self._csemap)  # constraints that support reification
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]), csemap=self._csemap)  # supports >, <, !=
        cpm_cons = linearize_reified_variables(cpm_cons, min_values=2, csemap=self._csemap)
        cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
        cpm_cons = only_implies(cpm_cons, csemap=self._csemap)  # anything that can create full reif should go above...
        # COPT does not round towards zero, so no 'div' in supported set (consistent with the gurobi interface)
        cpm_cons = linearize_constraint(cpm_cons, supported=frozenset({"sum", "wsum","->","sub","min","max","mul","abs"}), csemap=self._csemap)  # the core of the MIP-linearization
        cpm_cons = only_positive_bv(cpm_cons, csemap=self._csemap)  # after linearization, rewrite ~bv into 1-bv
        return cpm_cons

    def add(self, cpm_expr_orig):
      """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: self
      """
      from coptpy import COPT

      # add new user vars to the set
      get_variables(cpm_expr_orig, collect=self.user_vars)

      # transform and post the constraints
      for cpm_expr in self.transform(cpm_expr_orig):
          self._add_transformed(cpm_expr)

      return self

    __add__ = add  # avoid redirect in superclass

    def _add_transformed(self, cpm_expr):
        """Post a single already-transformed constraint to the COPT model. Returns the COPT constraint. Also used in `mus_native` to post transformed CPMpy constraints and gain access to the COPT constraint."""
        from coptpy import COPT

        # Comparisons: only numeric ones as 'only_implies()' has removed the '==' reification for Boolean expressions
        # numexpr `comp` bvar|const
        if isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            coptrhs = self.solver_var(rhs)

            # Thanks to `only_numexpr_equality()` only supported comparisons should remain
            if cpm_expr.name == '<=':
                coptlhs = self._make_numexpr(lhs)
                return self.copt_model.addConstr(coptlhs, COPT.LESS_EQUAL, coptrhs)
            elif cpm_expr.name == '>=':
                coptlhs = self._make_numexpr(lhs)
                return self.copt_model.addConstr(coptlhs, COPT.GREATER_EQUAL, coptrhs)
            elif cpm_expr.name == '==':
                if isinstance(lhs, _NumVarImpl) \
                        or (isinstance(lhs, Operator) and (lhs.name == 'sum' or lhs.name == 'wsum' or lhs.name == "sub")):
                    # a BoundedLinearExpression LHS, special case, like in objective
                    coptlhs = self._make_numexpr(lhs)
                    return self.copt_model.addConstr(coptlhs, COPT.EQUAL, coptrhs)

                elif lhs.name == 'mul':
                    assert len(lhs.args) == 2, "COPT only supports multiplication with 2 variables"
                    a, b = self.solver_vars(lhs.args)
                    return self.copt_model.addQConstr(a * b, COPT.EQUAL, coptrhs)

                elif lhs.name == 'div':
                    if not is_num(lhs.args[1]):
                        raise NotSupportedError(f"COPT only supports division by constants, but got {lhs.args[1]}")
                    a, b = self.solver_vars(lhs.args)
                    return self.copt_model.addConstr(a / b, COPT.EQUAL, coptrhs)

                else:
                    # General constraints; the rhs must be a COPT variable for these APIs.
                    if is_num(coptrhs):
                        coptrhs = self.solver_var(intvar(lb=coptrhs, ub=coptrhs))

                    if lhs.name == 'min':
                        return self.copt_model.addGenConstrMin(coptrhs, self.solver_vars(lhs.args))
                    elif lhs.name == 'max':
                        return self.copt_model.addGenConstrMax(coptrhs, self.solver_vars(lhs.args))
                    elif lhs.name == 'abs':
                        # COPT's addGenConstrAbs takes (resvar, argvar) where argvar must be a Var
                        arg = lhs.args[0]
                        if not isinstance(arg, _NumVarImpl):
                            raise NotSupportedError(f"COPT abs expects a variable argument, got {arg}")
                        return self.copt_model.addGenConstrAbs(coptrhs, self.solver_var(arg))
                    else:
                        raise NotImplementedError(
                        "Not a known supported copt comparison '{}' {}".format(lhs.name, cpm_expr))
            else:
                raise NotImplementedError(
                "Not a known supported copt comparison '{}' {}".format(lhs.name, cpm_expr))

        elif isinstance(cpm_expr, Operator) and cpm_expr.name == "->":
            # Indicator constraints
            # Take form bvar -> sum(x,y,z) >= rvar
            cond, sub_expr = cpm_expr.args
            assert isinstance(cond, _BoolVarImpl), f"Implication constraint {cpm_expr} must have BoolVar as lhs"
            assert isinstance(sub_expr, Comparison), "Implication must have linear constraints on right hand side"
            if isinstance(cond, NegBoolView):
                cond_var, bool_val = cond._bv, False
            else:
                cond_var, bool_val = cond, True
            cond_solver = self.solver_var(cond_var)

            lhs, rhs = sub_expr.args
            if not (isinstance(lhs, _NumVarImpl) or lhs.name == "sum" or lhs.name == "wsum"):
                raise Exception(f"Unknown linear expression {lhs} on right side of indicator constraint: {cpm_expr}")

            # COPT's indicator constraint silently fails to enforce the linear
            # constraint when the indicator's binary variable also appears in
            # the linear expression (a known COPT limitation). To avoid this,
            # strip the binvar's contribution from the lhs and fold it into
            # the rhs as a constant (binvar is fixed to `bool_val` when active).
            if isinstance(lhs, _NumVarImpl):
                terms = [(1, lhs)]
            elif lhs.name == "sum":
                terms = [(1, v) for v in lhs.args]
            else:  # wsum
                terms = list(zip(*lhs.args))
            cond_coef = sum(w for w, v in terms if v is cond_var)
            kept = [(w, v) for w, v in terms if v is not cond_var]

            if cond_coef == 0 or not is_num(rhs):
                # No binvar in lhs, or rhs is a variable we cannot shift by a constant.
                lin_lhs = self._make_numexpr(lhs)
                copt_rhs = self.solver_var(rhs)
            else:
                new_rhs = argval(rhs) - cond_coef * bool_val
                if kept:
                    lin_lhs = coptpy.quicksum(w * self.solver_var(v) for w, v in kept)
                else:
                    lin_lhs = 0
                copt_rhs = new_rhs

            if sub_expr.name == "<=":
                builder = lin_lhs <= copt_rhs
            elif sub_expr.name == ">=":
                builder = lin_lhs >= copt_rhs
            elif sub_expr.name == "==":
                builder = lin_lhs == copt_rhs
            else:
                raise Exception(f"Unknown linear expression {sub_expr} name")
            return self.copt_model.addGenConstrIndicator(cond_solver, bool_val, builder)

        # True or False
        elif isinstance(cpm_expr, BoolVal):
            # COPT's addConstr accepts a python bool through its overload.
            return self.copt_model.addConstr(cpm_expr.args[0])

        # a direct constraint, pass to solver
        elif isinstance(cpm_expr, DirectConstraint):
            cpm_expr.callSolver(self, self.copt_model)

        else:
            raise NotImplementedError(cpm_expr)  # if you reach this... please report on github

    def solution_hint(self, cpm_vars:List[_NumVarImpl], vals:List[int|bool]):
        """
        COPT supports warmstarting the solver with a (in)feasible solution through MIP starts.
        The provided value will affect branching heuristics during solving, making it more likely the final solution will contain the provided assignment.

        To learn more about MIP starts in COPT, see the 'MIP Starts' section of the COPT user guide.

        :param cpm_vars: list of CPMpy variables
        :param vals: list of (corresponding) values for the variables
        """
        # Flatten nested lists to handle test cases like solution_hint([a,[b]], [[[False]], True])
        flat_vars = []
        flat_vals = []
        def _flatten(vs, vls):
            for v, val in zip(vs, vls):
                if is_any_list(v) or is_any_list(val):
                    _flatten(v if is_any_list(v) else [v], val if is_any_list(val) else [val])
                else:
                    flat_vars.append(v)
                    flat_vals.append(val)
        _flatten(cpm_vars, vals)

        self.copt_model.setMipStart(self.solver_vars(flat_vars), [float(v) for v in flat_vals])
        self.copt_model.loadMipStart()

    @classmethod
    def mus_native(cls, soft, hard=[]):
        """
        Compute a MUS using COPT's native IIS (Irreducible Inconsistent Subsystem) algorithm.

        The main 'difficulty' is that COPT's native IIS algorithm expects individual constraints,
        while CPMpy always takes a 'grouped' perspective (e.g. one soft constraint can be a conjunction,
        or it can be a global that is decomposed/rewritten into multiple constraints).

        The code takes care to leave soft constraints corresponding to a single COPT constraint as-is,
        and adds a new 01 variable plus an implication/'indicator' constraint for each constraint in the group.

        Args:
            soft: List of soft constraints over which a MUS needs to be found
            hard: List of hard constraints that always need to be satisfied

        Returns a MUS (list of constraints from soft that is unsatisfiable together, and subset minimal).
        """

        soft_cons = toplevel_list(soft, merge_and=False)

        # instantiate COPT solver
        s = cls()

        # collect the COPT constraint objects
        copt_hard_cons = []
        copt_soft_cons = []

        for soft_con in soft_cons:
            # transform each constraint separately, can map to multiple COPT-level constraints
            soft_con_tf = s.transform(soft_con)

            if len(soft_con_tf) == 0:
                # uncommon case, just ensure `copt_soft_con` and `soft` are same length
                soft_con_tf = [cp.BoolVal(True)]

            if len(soft_con_tf) == 1:
                # if `con` represented by a single transformed constraint, it can be added as-is
                soft_con_rep = soft_con_tf[0]
                copt_soft_cons.append(s._add_transformed(soft_con_rep))
            else:
                # We introduce an assumption variable `a` and add *hard* constraints `a -> /\ tf_cons`.
                # The lower bound fixing `a == 1` is the soft part COPT can select in the IIS.
                assumption = cp.boolvar()

                # add `a -> /\ C` as a hard constraint
                hard.append(assumption.implies(cp.all(soft_con_tf)))

                soft_con_rep = s.solver_var(assumption)
                soft_con_rep.lb = 1
                copt_soft_cons.append(soft_con_rep)



        # transform and add all hard constraints
        for cpm_con in s.transform(hard):
            # use ._add_transformed instead of .add because we need the COPT constraint object later
            copt_hard_cons.append(s._add_transformed(cpm_con))

        # update model so we can access constraint attributes
        # model updates can be expensive, so we do this only once!
        s.native_model.update()
        for copt_con in copt_hard_cons:
            # Force the constraint to be in the IIS; COPT does not expose a per-constraint
            # 'force' attribute, so we rely on computeIIS considering all constraints by default.
            pass

        # compute IIS (conveniently fails if original model was SAT since it will solve the model)
        try:
            s.native_model.computeIIS()
        except Exception as e:
            # COPT raises a CoptError when the model is not infeasible; treat that as a misuse.
            raise AssertionError("MUS: model must be UNSAT") from e

        def in_iis(copt_con):
            """Check if `copt_con` is in the IIS. COPT distinguishes lower/upper bound participation."""
            if isinstance(copt_con, coptpy.Var):
                return s.native_model.getVarLowerIIS(copt_con) or s.native_model.getVarUpperIIS(copt_con)
            else:
                return s.native_model.getConstrLowerIIS(copt_con) or s.native_model.getConstrUpperIIS(copt_con)

        # Return `soft_con` if its representing COPT constraint is in the IIS
        return [soft_con for soft_con, copt_soft_con in zip(soft_cons, copt_soft_cons) if in_iis(copt_soft_con)]

    def _get_callback(self, display:Callback, events:Iterable) -> Callable:
        """
        Get the callback function to use for COPT.
        Arguments:
            display: either an expression, a list of expressions, or a callback function
            events: iterable of COPT callback context codes (e.g. COPT.CBCONTEXT_MIPSOL)
        """

        import coptpy
        from coptpy import COPT

        self.events = frozenset(events)
        if isinstance(display, Expression) or is_any_list(display):
            cpm_vars = get_variables(display)
        else:
            cpm_vars = list(self.user_vars)
        copt_vars = coptpy.VarArray()
        for v in self.solver_vars(cpm_vars):
            copt_vars.pushBack(v)

        outer = self

        class _MIPSolCallback(coptpy.CallbackBase):
            def callback(self):
                if self.where() not in outer.events:
                    return # irrelevant event
                # COPT exposes the incumbent solution through getIncumbent / getSolution
                try:
                    sol_vals = self.getSolution(copt_vars)
                except Exception:
                    return
                for cpm_var, solver_val in zip(cpm_vars, sol_vals):
                    if cpm_var.is_bool():
                        cpm_var._value = solver_val >= 0.5
                    else:
                        cpm_var._value = int(solver_val)

                outer.print_display(display)

        return _MIPSolCallback()
