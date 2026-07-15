#!/usr/bin/env python3
"""Tests for the direct-API GAMS solver backend."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import GREATER_EQUAL, Model, solvers
from linopy.constants import TerminationCondition
from linopy.solvers import GAMS, _iter_sos_sets

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
