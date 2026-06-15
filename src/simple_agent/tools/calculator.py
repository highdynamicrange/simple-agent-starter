import ast
import math
import operator

from pydantic import BaseModel, Field


class CalculatorArguments(BaseModel):
    expression: str = Field(min_length=1, max_length=200)


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_NODES = 50
_MAX_ABS_RESULT = 1_000_000_000_000_000
_MAX_ABS_EXPONENT = 10


def calculate(arguments: CalculatorArguments) -> dict[str, int | float]:
    try:
        tree = ast.parse(arguments.expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("表达式语法无效。") from exc

    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise ValueError("表达式过于复杂。")

    result = _evaluate(tree.body)
    if isinstance(result, bool) or not isinstance(result, int | float):
        raise ValueError("表达式结果必须是数字。")
    if not math.isfinite(result) or abs(result) > _MAX_ABS_RESULT:
        raise ValueError("计算结果超出允许范围。")
    return {"result": result}


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ValueError("只允许数字常量。")
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        result = _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
        return _check_intermediate(result)

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_ABS_EXPONENT:
            raise ValueError("指数绝对值不能超过 10。")
        try:
            result = _BINARY_OPERATORS[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise ValueError("不能除以零。") from exc
        except OverflowError as exc:
            raise ValueError("计算结果超出允许范围。") from exc
        return _check_intermediate(result)

    raise ValueError("表达式包含不允许的语法。")


def _check_intermediate(value: int | float) -> int | float:
    if not math.isfinite(value) or abs(value) > _MAX_ABS_RESULT:
        raise ValueError("计算结果超出允许范围。")
    return value
