from __future__ import annotations

import pytest

from rene.operation import (
    CandidateExecutionOptions,
    DummyOperation,
    ExecutionOption,
    OperationSpec,
    Operation,
)


@pytest.fixture
def mock_spec() -> OperationSpec:
    unit_time = 1.0
    spec = OperationSpec(
        options=CandidateExecutionOptions(
            options=[
                ExecutionOption[str](
                    real_time=123.0,
                    unit_time=unit_time,
                    cost=3.0,
                    knob="one",
                ),
                ExecutionOption[str](
                    real_time=789.0,
                    unit_time=unit_time,
                    cost=1.0,
                    knob="three",
                ),
                ExecutionOption[str](
                    real_time=456.0,
                    unit_time=unit_time,
                    cost=2.0,
                    knob="two",
                ),
                ExecutionOption[str](
                    real_time=234.0,
                    unit_time=unit_time,
                    cost=3.1,
                    knob="four",
                ),
            ]
        ),
        cost_model=None,  # type: ignore
    )
    return spec


def test_operation_spec_option_filtering(mock_spec: OperationSpec) -> None:
    assert len(mock_spec.options.options) == 3
    assert set(o.knob for o in mock_spec.options.options) == set(
        ["one", "two", "three"]
    )
    assert set(o.quant_time for o in mock_spec.options.options) == set([123, 456, 789])


def test_dummy_operation() -> None:
    op = DummyOperation()
    with pytest.raises(AttributeError):
        op.spec


def test_operation_computed_fields(mock_spec: OperationSpec) -> None:
    op = Operation(spec=mock_spec)
    assert op.min_duration == 123
    assert op.max_duration == 789
    assert not op.is_dummy


def test_knob_assignment(mock_spec: OperationSpec) -> None:
    def op_assign_assert(duration: int, knob) -> None:
        op = Operation(spec=mock_spec)
        op.duration = duration
        op.assign_knob()
        assert op.assigned_knob == knob

    with pytest.raises(ValueError, match=r".*20.*[123, 789]"):
        op_assign_assert(20, None)
    with pytest.raises(ValueError, match=r".*122.*[123, 789]"):
        op_assign_assert(122, None)

    op_assign_assert(123, "one")
    op_assign_assert(124, "one")
    op_assign_assert(234, "one")
    op_assign_assert(455, "one")
    op_assign_assert(456, "two")
    op_assign_assert(457, "two")
    op_assign_assert(458, "two")
    op_assign_assert(654, "two")
    op_assign_assert(788, "two")
    op_assign_assert(789, "three")

    with pytest.raises(ValueError, match=r".*790.*[123, 789]"):
        op_assign_assert(790, None)
    with pytest.raises(ValueError, match=r".*1234.*[123, 789]"):
        op_assign_assert(1234, None)
