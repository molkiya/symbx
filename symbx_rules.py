# symbx_rules.py
from dataclasses import dataclass
from typing import Callable, List
import hashlib

@dataclass
class Rule:
    name: str
    family: str
    parametric: bool
    fn: Callable[[float, dict], float]  # (x, params) -> x'

def add_rule(k: float) -> Rule:
    return Rule(name=f"+{k}", family="add", parametric=False, fn=lambda x, _: x + k)

def mul_rule(k: float) -> Rule:
    return Rule(name=f"*{k}", family="mul", parametric=False, fn=lambda x, _: x * k)

RULES: List[Rule] = [
    add_rule(1.0), add_rule(2.0), add_rule(3.0),
    mul_rule(2.0), mul_rule(3.0),
]

NAME2RULE = {r.name: r for r in RULES}

def execute_program(a_value: float, prog_names: List[str]) -> float:
    x = a_value
    for name in prog_names:
        r = NAME2RULE[name]
        x = r.fn(x, {})
    return x

def canonical_form(prog_names: List[str]) -> str:
    # simple canonical form: space-joined tokens; can be upgraded to S-expr
    return " ".join(prog_names)

def program_hash(canon: str) -> str:
    # sha256 over canonical form
    h = hashlib.sha256()
    h.update(canon.encode("utf-8"))
    return h.hexdigest()
