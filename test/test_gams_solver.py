#!/usr/bin/env python3
"""Tests for the direct-API GAMS solver backend."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import GREATER_EQUAL, Model, Variable, solvers
from linopy.constants import TerminationCondition
from linopy.solvers import (
    GAMS,
    _iter_sos_sets,
    _names_to_labels,
    _truncate_for_gams_uel,
)

pytestmark = pytest.mark.skipif(
    "gams" not in solvers.licensed_solvers, reason="GAMS/gamspy license not available"
)


@pytest.fixture
def simple_model() -> Model:
    m = Model(chunk=None)
    x = m.add_variables(name="x")
    y = m.add_variables(name="y")
    m.add_constraints(2 * x + 6 * y, GREATER_EQUAL, 10)
    m.add_constraints(4 * x + 2 * y, GREATER_EQUAL, 3)
    m.add_objective(2 * y + x)
    return m


def test_solve_kwargs_partitions_options() -> None:
    solver = GAMS(options={"solver": "CPLEX", "time_limit": 10, "made_up_option": 1})
    kwargs = solver._solve_kwargs({"time_limit", "iteration_limit"})

    assert kwargs["solver"] == "CPLEX"
    assert kwargs["options"].time_limit == 10
    assert kwargs["solver_options"] == {"made_up_option": 1}


def test_solve_kwargs_omits_empty_partitions() -> None:
    solver = GAMS(options={})
    kwargs = solver._solve_kwargs({"time_limit"})

    assert "options" not in kwargs
    assert "solver_options" not in kwargs
    assert "solver" not in kwargs


@pytest.mark.parametrize(
    ("solve_stat", "model_stat", "expected"),
    [
        (1, 1, TerminationCondition.optimal),
        (1, 4, TerminationCondition.infeasible),
        (1, 3, TerminationCondition.unbounded),
        (2, 1, TerminationCondition.iteration_limit),
        (3, 1, TerminationCondition.time_limit),
        (8, 1, TerminationCondition.user_interrupt),
        (9, 1, TerminationCondition.internal_solver_error),
        (1, 11, TerminationCondition.licensing_problems),
        (1, 999, TerminationCondition.unknown),
    ],
)
def test_status_mapping(solve_stat: int, model_stat: int, expected: str) -> None:
    solver = GAMS()
    termination = solver._SOLVE_STAT_MAP.get(
        solve_stat, solver._MODEL_STAT_MAP.get(model_stat, "unknown")
    )
    assert termination == expected


def test_build_direct_creates_expected_symbols(simple_model: Model) -> None:
    solver = GAMS.from_model(simple_model, io_api="direct")
    cont = solver.solver_model.container

    assert cont["j"].records.shape[0] == 2
    assert cont["jc"].records.shape[0] == 2
    assert "xc" in cont.data
    assert "eg" in cont.data


def test_solve_lp(simple_model: Model) -> None:
    status, condition = simple_model.solve("gams")
    assert status == "ok"
    assert condition == "optimal"
    assert simple_model.objective.value == pytest.approx(3.3)


def test_solve_mip() -> None:
    m = Model(chunk=None)
    b = m.add_variables(lower=0, upper=5, integer=True, name="b")
    y = m.add_variables(lower=0, upper=1, binary=True, name="y")
    x = m.add_variables(lower=-3, upper=10, name="x")
    m.add_constraints(x + b >= 2, name="c1")
    m.add_constraints(x - y == 1, name="c2")
    m.add_objective(x + b + y, sense="min")

    status, condition = m.solve("gams")
    assert status == "ok"
    assert condition == "optimal"
    assert m.objective.value == pytest.approx(2.0)
    assert float(y.solution) in (0.0, 1.0)
    assert float(b.solution) == pytest.approx(round(float(b.solution)))


def test_solve_sos1() -> None:
    m = Model(chunk=None)
    idx = pd.Index([0, 1, 2, 3], name="i")
    sos_var = m.add_variables(lower=0, upper=1, coords=[idx], name="sos_var")
    m.add_sos_constraints(sos_var, sos_type=1, sos_dim="i")
    coefs = xr.DataArray([-1.0, -2.0, -3.0, -4.0], coords=[idx], dims=["i"])
    m.add_objective((sos_var * coefs).sum())

    assert list(_iter_sos_sets(m))  # sanity: the model does have a SOS group

    status, condition = m.solve("gams")
    assert status == "ok"
    assert condition == "optimal"
    # SOS1: only the single largest-magnitude coefficient entry may be nonzero.
    assert m.objective.value == pytest.approx(-4.0)
    solution = sos_var.solution.values
    assert np.count_nonzero(solution) == 1
    assert solution[3] == pytest.approx(1.0)


def test_solve_sos2() -> None:
    m = Model(chunk=None)
    idx = pd.Index([0, 1, 2, 3], name="i")
    sos_var = m.add_variables(lower=0, upper=1, coords=[idx], name="sos_var")
    m.add_sos_constraints(sos_var, sos_type=2, sos_dim="i")
    coefs = xr.DataArray([-1.0, -2.0, -3.0, -4.0], coords=[idx], dims=["i"])
    m.add_objective((sos_var * coefs).sum())

    status, condition = m.solve("gams")
    assert status == "ok"
    assert condition == "optimal"
    # SOS2 permits two *adjacent* nonzero entries: the best pair is the last
    # two (coefficients -3, -4), both driven to their upper bound of 1.
    assert m.objective.value == pytest.approx(-7.0)
    solution = sos_var.solution.values
    assert np.count_nonzero(solution) == 2
    assert solution[2] == pytest.approx(1.0)
    assert solution[3] == pytest.approx(1.0)


def test_solve_qp() -> None:
    m = Model(chunk=None)
    lower = pd.Series(0, range(3))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(lower, name="y")
    m.add_constraints(x + y >= 10)
    m.add_objective(x * x - 2 * x + y)

    solver = GAMS.from_model(m, io_api="direct")
    assert solver.solver_model.problem.name.lower() == "qcp"

    status, condition = m.solve("gams")
    assert status == "ok"
    assert condition == "optimal"
    assert m.objective.value == pytest.approx(23.25)
    assert x.solution.values == pytest.approx([1.5, 1.5, 1.5])
    assert y.solution.values == pytest.approx([8.5, 8.5, 8.5])


def test_solve_miqp() -> None:
    m = Model(chunk=None)
    idx = pd.RangeIndex(3, name="i")
    x = m.add_variables(coords=[idx], name="x", binary=True)
    y = m.add_variables(lower=0, coords=[idx], name="y")
    m.add_constraints(x + y >= 1)
    m.add_objective((x * x - 3 * x + y).sum())

    solver = GAMS.from_model(m, io_api="direct")
    assert solver.solver_model.problem.name.lower() == "miqcp"

    status, condition = m.solve("gams")
    assert status == "ok"
    assert condition == "optimal"
    assert m.objective.value == pytest.approx(-6.0)
    assert (x.solution.values == 1.0).all()
    assert (y.solution.values == 0.0).all()


def test_solve_sos1_with_quadratic_objective() -> None:
    m = Model(chunk=None)
    idx = pd.Index([0, 1, 2], name="i")
    x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
    m.add_sos_constraints(x, sos_type=1, sos_dim="i")
    a = xr.DataArray([0.1, 0.2, 5.0], coords=[idx], dims=["i"])
    # x^2 - 2*a*x is minimized (unconstrained) at x=a; SOS1 forces all but
    # one entry to 0, so the optimum picks whichever single index reduces
    # the objective the most -- here index 2 (a=5, clipped to the upper
    # bound of 1) by a wide, untied margin.
    m.add_objective((x * x - 2 * a * x).sum())

    solver = GAMS.from_model(m, io_api="direct")
    assert solver.solver_model.problem.name.lower() == "miqcp"

    status, condition = m.solve("gams")
    assert status == "ok"
    # GAMS reports MINLP-class incumbents (model status 8, "Integer
    # Solution") as "suboptimal" even when the B&B gap closes to zero --
    # it never claims global optimality for a nonlinear-branching solve.
    # The GAMS log for this solve shows "Best possible: -9.0000" exactly
    # matching the incumbent with a 0.000000 gap, confirming the answer
    # below is in fact exactly optimal despite the reported condition.
    assert condition == "suboptimal"
    assert m.objective.value == pytest.approx(-9.0)
    solution = x.solution.values
    assert np.count_nonzero(solution) == 1
    assert solution[2] == pytest.approx(1.0)


def test_solve_sos2_with_quadratic_objective() -> None:
    m = Model(chunk=None)
    idx = pd.Index([0, 1, 2, 3], name="i")
    x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
    m.add_sos_constraints(x, sos_type=2, sos_dim="i")
    a = xr.DataArray([6.0, 0.1, 0.1, 5.0], coords=[idx], dims=["i"])
    # SOS2 permits two *adjacent* nonzero entries. The two largest `a`
    # values (indices 0 and 3) are not adjacent, so this also exercises
    # that the adjacency rule -- not just "pick the two best" -- is
    # actually enforced: the best *adjacent* pair is (0, 1).
    m.add_objective((x * x - 2 * a * x).sum())

    status, condition = m.solve("gams")
    assert status == "ok"
    # Same GAMS MINLP-status convention as test_solve_sos1_with_quadratic_objective.
    assert condition == "suboptimal"
    assert m.objective.value == pytest.approx(-11.01)
    solution = x.solution.values
    assert np.count_nonzero(solution) == 2
    assert solution[0] == pytest.approx(1.0)
    assert solution[1] == pytest.approx(0.1)
    assert solution[2] == pytest.approx(0.0)
    assert solution[3] == pytest.approx(0.0)


def test_solve_combined_sos1_sos2_continuous_with_quadratic_objective() -> None:
    m = Model(chunk=None)

    idx1 = pd.Index([0, 1, 2], name="i1")
    x1 = m.add_variables(lower=0, upper=1, coords=[idx1], name="x1")
    m.add_sos_constraints(x1, sos_type=1, sos_dim="i1")
    a1 = xr.DataArray([0.1, 0.2, 5.0], coords=[idx1], dims=["i1"])

    idx2 = pd.Index([0, 1, 2, 3], name="i2")
    x2 = m.add_variables(lower=0, upper=1, coords=[idx2], name="x2")
    m.add_sos_constraints(x2, sos_type=2, sos_dim="i2")
    a2 = xr.DataArray([6.0, 0.1, 0.1, 5.0], coords=[idx2], dims=["i2"])

    y = m.add_variables(lower=0, upper=2, name="y")
    # Non-binding at the optimum (x1 sums to 1 and y to 2), but exercises
    # SOS + plain-continuous + quadratic all contributing to one constraint.
    m.add_constraints(x1.sum() + y >= 1, name="link")

    m.add_objective(
        (x1 * x1 - 2 * a1 * x1).sum()
        + (x2 * x2 - 2 * a2 * x2).sum()
        + (y * y - 2 * 3 * y)
    )

    solver = GAMS.from_model(m, io_api="direct")
    assert solver.solver_model.problem.name.lower() == "miqcp"

    status, condition = m.solve("gams")
    assert status == "ok"
    # Same GAMS MINLP-status convention as test_solve_sos1_with_quadratic_objective.
    assert condition == "suboptimal"
    # The objective separates additively across x1/x2/y with only a
    # non-binding shared constraint, so the global optimum is the sum of
    # each group's independent optimum computed in the tests above
    # (-9.0, -11.01) plus y clipped to its upper bound of 2: 4 - 12 = -8.
    assert m.objective.value == pytest.approx(-9.0 - 11.01 - 8.0)

    sol1 = x1.solution.values
    assert np.count_nonzero(sol1) == 1
    assert sol1[2] == pytest.approx(1.0)

    sol2 = x2.solution.values
    assert np.count_nonzero(sol2) == 2
    assert sol2[0] == pytest.approx(1.0)
    assert sol2[1] == pytest.approx(0.1)

    assert float(y.solution) == pytest.approx(2.0)


def test_sos_quadratic_linking_equations_reference_only_own_member(
    tmp_path: Path,
) -> None:
    """
    Regression guard for the exact bug this feature's design caught: a
    naive ``Sum(dom[s_set, j], ...)`` linking equation summed every row
    over the *entire* SOS group instead of each row's own member. Checking
    only final solved values could pass by coincidence on a symmetric toy
    model, so this inspects the actual compiled GAMS equation text.
    """
    m = Model(chunk=None)
    idx = pd.Index([0, 1], name="i")
    x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
    m.add_sos_constraints(x, sos_type=1, sos_dim="i")
    m.add_objective((x * x).sum())

    listing_fn = tmp_path / "listing.lst"
    m.solve(
        "gams",
        listing_file=str(listing_fn),
        variable_listing_limit=-1,
        equation_listing_limit=-1,
    )
    listing = listing_fn.read_text()

    rows = re.findall(r"elink_xs1\(j\d+\)\.\.(.*?);", listing)
    assert rows, "elink_xs1 equations not found in the GAMS listing"
    for row in rows:
        # Each row must reference exactly one xs1(...) term -- its own
        # position's SOS member, not the whole group.
        assert row.count("xs1(") == 1, f"row references more than one member: {row}"


def test_truncate_for_gams_uel_preserves_suffix_and_uniqueness() -> None:
    long_prefix = "a" * 80
    names = [f"{long_prefix}#0", f"{long_prefix}#1"]

    truncated = _truncate_for_gams_uel(names, limit=63)

    assert all(len(n) <= 63 for n in truncated)
    assert truncated[0].endswith("#0")
    assert truncated[1].endswith("#1")
    assert truncated[0] != truncated[1]
    # Round-trips to the correct labels even though the prefixes collide.
    assert _names_to_labels(truncated).tolist() == [0, 1]


def test_truncate_for_gams_uel_leaves_short_names_untouched() -> None:
    names = ["short#0", "j1"]
    assert _truncate_for_gams_uel(names, limit=63) == names


def test_build_direct_default_naming_unaffected_by_explicit_coordinate_names_support(
    simple_model: Model,
) -> None:
    """The default (off) path's Set records must stay exactly `j{k}`/`i{k}`."""
    solver = GAMS.from_model(simple_model, io_api="direct")
    cont = solver.solver_model.container
    assert cont["j"].records["uni"].tolist() == ["j0", "j1"]


def test_solve_with_explicit_coordinate_names() -> None:
    m = Model(chunk=None)
    idx = pd.Index(["plantA", "plantB", "plantC"], name="plant")
    x = m.add_variables(lower=0, upper=10, coords=[idx], name="production")
    m.add_constraints(x.sum() >= 5, name="demand")
    m.add_constraints(x >= 1, name="minrun")
    m.add_objective((x * 2).sum())

    solver = GAMS.from_model(m, io_api="direct", explicit_coordinate_names=True)
    cont = solver.solver_model.container
    j_keys = cont["j"].records["uni"].tolist()
    assert any("production" in key for key in j_keys)
    assert j_keys != ["j0", "j1", "j2"]

    status, condition = m.solve("gams", explicit_coordinate_names=True)
    assert status == "ok"
    assert condition == "optimal"
    assert m.objective.value == pytest.approx(10.0)
    assert x.solution.to_pandas().to_dict() == {
        "plantA": 3.0,
        "plantB": 1.0,
        "plantC": 1.0,
    }


def test_solve_with_explicit_coordinate_names_matches_default() -> None:
    def build() -> tuple[Model, Variable]:
        m = Model(chunk=None)
        idx = pd.Index(["plantA", "plantB", "plantC"], name="plant")
        x = m.add_variables(lower=0, upper=10, coords=[idx], name="production")
        m.add_constraints(x.sum() >= 5, name="demand")
        m.add_constraints(x >= 1, name="minrun")
        m.add_objective((x * 2).sum())
        return m, x

    m_off, x_off = build()
    m_off.solve("gams")
    m_on, x_on = build()
    m_on.solve("gams", explicit_coordinate_names=True)

    assert m_off.objective.value == pytest.approx(m_on.objective.value)
    assert x_off.solution.values == pytest.approx(x_on.solution.values)


def test_solve_with_explicit_coordinate_names_truncates_long_coordinates() -> None:
    """Names exceeding GAMS's 63-char UEL limit must still solve correctly."""
    long_names = [f"a_very_long_descriptive_coordinate_name_{i}" * 3 for i in range(3)]
    idx = pd.Index(long_names, name="plant")

    m = Model(chunk=None)
    x = m.add_variables(lower=0, upper=10, coords=[idx], name="production")
    m.add_constraints(x.sum() >= 5, name="demand")
    m.add_constraints(x >= 1, name="minrun")
    m.add_objective((x * 2).sum())

    status, condition = m.solve("gams", explicit_coordinate_names=True)
    assert status == "ok"
    assert condition == "optimal"
    assert m.objective.value == pytest.approx(10.0)
