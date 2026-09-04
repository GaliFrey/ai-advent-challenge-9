"""Restricted Python subset for the built-in exercise; launched without app secrets."""

import ast
import json
import resource
import sys


ALLOWED = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
    ast.If, ast.For, ast.Expr, ast.Assert, ast.Lambda, ast.Call, ast.keyword,
    ast.Name, ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple,
    ast.Subscript, ast.Slice, ast.Attribute, ast.ListComp, ast.comprehension,
    ast.Compare, ast.Eq, ast.NotEq, ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.Pass,
}
BUILTINS = {"tuple": tuple, "list": list, "len": len, "range": range, "enumerate": enumerate}


def restricted_tree(source, *, function):
    if len(source) > 16000:
        raise ValueError("Код превышает лимит проверки")
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    if len(nodes) > 2000 or any(type(node) not in ALLOWED for node in nodes):
        raise ValueError("Конструкция Python вне поддерживаемого подмножества")
    for node in nodes:
        if isinstance(node, (ast.Name, ast.arg)) and (node.id if isinstance(node, ast.Name) else node.arg).startswith("_"):
            raise ValueError("Служебные имена запрещены")
        if isinstance(node, ast.Attribute) and (node.attr != "append" or not isinstance(node.ctx, ast.Load)):
            raise ValueError("Разрешён только метод append")
        if isinstance(node, ast.FunctionDef) and (node is not tree.body[0] or not function or node.name != "build" or node.decorator_list or node.returns):
            raise ValueError("Разрешено только определение build без декораторов и аннотаций")
        if isinstance(node, ast.arg) and node.annotation:
            raise ValueError("Аннотации не поддерживаются проверкой")
        if isinstance(node, ast.Assert) and node not in tree.body:
            raise ValueError("Проверяются только assert верхнего уровня")
    if function and (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)):
        raise ValueError("В разделе исправления ожидается только функция build")
    return tree


def result(ok, detail):
    return {"status": "pass" if ok else "fail", "detail": detail}


def check_contract(tree):
    args = tree.body[0].args
    signature = [a.arg for a in args.args] == ["values", "history"] and len(args.defaults) == 1 and not (args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg)
    checks = {"signature": result(signature, "build(values, history=...) с необязательной историей")}

    def fresh():
        namespace = {"__builtins__": dict(BUILTINS)}
        exec(compile(tree, "<build>", "exec"), namespace)
        return namespace["build"]

    def independent(build):
        first, second = build([1, 2]), build([3])
        return [h() for h in first], [h() for h in second]

    def explicit_none(build):
        first, second = build([4], None), build([5], None)
        return first[0](), second[0]()

    def shared_history(build):
        shared = [0]
        first, second = build([1, 2], shared), build([3], shared)
        before = shared[:]
        shared.append(99)
        shared[0] = -1
        return before, [h() for h in first], [h() for h in second]

    def empty(build):
        shared = [7]
        return build([]), build([], None), build([], shared), shared

    cases = [
        ("independent", independent, ([(1, (1,)), (2, (1, 2))], [(3, (3,))])),
        ("explicit_none", explicit_none, ((4, (4,)), (5, (5,)))),
        ("shared_snapshot", shared_history, ([0, 1, 2, 3], [(1, (0, 1)), (2, (0, 1, 2))], [(3, (0, 1, 2, 3))])),
        ("empty", empty, ([], [], [], [7])),
    ]
    for name, run, expected in cases:
        try:
            actual = run(fresh())
            checks[name] = result(repr(actual) == repr(expected), f"Получено: {actual!r}; ожидается: {expected!r}")
        except Exception as exc:
            checks[name] = result(False, type(exc).__name__)
    return {"status": "pass" if all(c["status"] == "pass" for c in checks.values()) else "fail", "cases": checks}


def check_asserts(function, source):
    tree = restricted_tree(source, function=False)
    count = sum(isinstance(node, ast.Assert) for node in tree.body)
    if not count:
        return {"status": "unknown", "detail": "Не найдены assert верхнего уровня"}
    namespace = {"__builtins__": dict(BUILTINS)}
    exec(compile(function, "<build>", "exec"), namespace)
    failures = []
    passed = 0
    for node in tree.body:
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<asserts>", "exec"), namespace)
            passed += isinstance(node, ast.Assert)
        except AssertionError:
            if not isinstance(node, ast.Assert):
                return {"status": "fail", "detail": "Ошибка подготовки assert"}
            failures.append(ast.unparse(node))
        except Exception as exc:
            return {"status": "fail", "detail": "Ошибка выполнения проверок: " + type(exc).__name__}
    return {"status": "fail" if failures else "pass", "passed": passed, "total": count, "failures": failures}


def verify(data):
    unknown = {"status": "unknown", "detail": "Не удалось проверить код"}
    try:
        function = restricted_tree(data["function"], function=True)
    except (ValueError, SyntaxError, RecursionError) as exc:
        return {"contract": {**unknown, "detail": str(exc)}, "asserts": unknown}
    contract = check_contract(function)
    try:
        assertions = check_asserts(function, data["asserts"])
    except (ValueError, SyntaxError, RecursionError) as exc:
        assertions = {**unknown, "detail": str(exc)}
    return {"contract": contract, "asserts": assertions}


if __name__ == "__main__":
    # Defense in depth: AST allowlist, no imports/introspection/I/O in evaluated code,
    # minimal builtins, fresh process/environment, bounded CPU/address space/output.
    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    print(json.dumps(verify(json.loads(sys.stdin.read(40000))), ensure_ascii=False))
