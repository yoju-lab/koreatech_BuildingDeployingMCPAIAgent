# servers/math_server.py
from fastmcp import FastMCP
import ast, operator as op
from pathlib import Path
from dotenv import load_dotenv

# --- .env 로드: 스크립트 상위 폴더의 .env를 명시적으로 찾고, 기존 env를 덮어쓰기 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

mcp = FastMCP("math")

# 허용 연산자만 매핑
OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
    ast.USub: op.neg, ast.UAdd: op.pos, ast.BitXor: op.xor  # 원치 않으면 제거
}

def _eval_ast(node):
    if isinstance(node, ast.Num):  # py<=3.7
        return node.n
    if isinstance(node, ast.Constant):  # py>=3.8
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("숫자만 허용됩니다.")
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        fn = OPS.get(type(node.op))
        if not fn:
            raise ValueError(f"허용되지 않은 연산자: {type(node.op).__name__}")
        return fn(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        fn = OPS.get(type(node.op))
        if not fn:
            raise ValueError(f"허용되지 않은 단항 연산자: {type(node.op).__name__}")
        return fn(operand)
    raise ValueError(f"허용되지 않은 표현식: {type(node).__name__}")

@mcp.tool
def eval(expression: str) -> dict:
    """
    안전한 수식 평가(사칙연산/거듭제곱/나머지/정수나눗셈).
    예) "3*(5+2) - 10/2"
    """
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_ast(tree.body)
        return {"value": value}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run()  # stdio
