import pytest
from pydantic import ValidationError

from simple_agent.tools.calculator import CalculatorArguments, calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(10 - 2) / 4", 2),
        ("2 ** 10", 1024),
        ("-5 + 2", -3),
    ],
)
def test_calculator(expression, expected):
    assert calculate(CalculatorArguments(expression=expression))["result"] == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo unsafe')",
        "sum([1, 2])",
        "value + 1",
        "True + 1",
    ],
)
def test_rejects_unsafe_syntax(expression):
    with pytest.raises(ValueError):
        calculate(CalculatorArguments(expression=expression))


def test_rejects_zero_division():
    with pytest.raises(ValueError, match="除以零"):
        calculate(CalculatorArguments(expression="1 / 0"))


def test_rejects_large_exponent():
    with pytest.raises(ValueError, match="指数"):
        calculate(CalculatorArguments(expression="2 ** 11"))


def test_rejects_long_expression():
    with pytest.raises(ValidationError):
        CalculatorArguments(expression="1" * 201)
