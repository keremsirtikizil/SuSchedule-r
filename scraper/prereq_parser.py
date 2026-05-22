"""
Parse a Sabancı University prereq_text string into a boolean expression tree.

Supported grammar (observed from the catalog):

  expr   := or_expr
  or_expr := and_expr ( 'or' and_expr )*
  and_expr := atom ( 'and' atom )*
  atom    := '(' expr ')' | course_clause
  course_clause := COURSE_CODE 'Undergraduate' '-' 'Min' 'Grade' GRADE
                   [ '(can be taken concurrently)' ]
  COURSE_CODE   := <SUBJ> <NUM>     e.g. CS 201, MAT 48005, CIP 101N
  GRADE         := single letter (always 'D' in current data)

Output tree:
  leaf:        {"course": "CS 300", "min_grade": "D", "concurrent": False}
  internal:    {"op": "and" | "or", "operands": [<expr>, ...]}

The function is forgiving: it ignores stray dashes and the literal placeholder
"__" returns None. On any unexpected input it raises ValueError so the caller
can decide whether to drop the course or surface a warning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

EMPTY_MARKERS = {"", "__"}

# Token types
TOK_LPAREN = "LPAREN"
TOK_RPAREN = "RPAREN"
TOK_AND = "AND"
TOK_OR = "OR"
TOK_ATOM = "ATOM"

_CODE_RE = re.compile(r"\b([A-Z]{2,5})\s+(\d{2,5}[A-Z]?)\b")
_ATOM_RE = re.compile(
    r"(?P<subj>[A-Z]{2,5})\s+(?P<num>\d{2,5}[A-Z]?)"
    r"\s*-\s*Undergraduate\s*-\s*Min\s*Grade\s*(?P<grade>[A-Z][+-]?)"
    r"(?P<concurrent>\s*\(\s*can\s+be\s+taken\s+concurrently\s*\))?",
    re.IGNORECASE,
)


@dataclass
class Token:
    kind: str
    value: Any = None  # for ATOM: dict with course/min_grade/concurrent


def tokenize(text: str) -> list[Token]:
    """Turn the raw prereq text into a list of tokens.

    We match atoms greedily, treat parentheses individually, and recognise
    the bare connectors 'and' / 'or' on word boundaries.
    """
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append(Token(TOK_LPAREN))
            i += 1
            continue
        if ch == ")":
            tokens.append(Token(TOK_RPAREN))
            i += 1
            continue
        # Word boundary 'and' / 'or'
        # Use a lookahead to be sure we don't eat 'and' inside a code (no codes
        # contain those letters in our data).
        if text[i:i + 4].lower() == "and " or text[i:i + 4].lower() == "and(":
            tokens.append(Token(TOK_AND))
            i += 3
            continue
        if text[i:i + 3].lower() == "or " or text[i:i + 3].lower() == "or(":
            tokens.append(Token(TOK_OR))
            i += 2
            continue
        # Try to match an atom starting here.
        m = _ATOM_RE.match(text, i)
        if m:
            atom = {
                "course": f"{m.group('subj').upper()} {m.group('num').upper()}",
                "min_grade": m.group("grade").upper(),
                "concurrent": bool(m.group("concurrent")),
            }
            tokens.append(Token(TOK_ATOM, atom))
            i = m.end()
            continue
        # Stray separator we can ignore
        if ch in "-,":
            i += 1
            continue
        raise ValueError(f"Unrecognised input at position {i}: {text[i:i + 40]!r}")
    return tokens


class _Parser:
    def __init__(self, tokens: list[Token]):
        self.toks = tokens
        self.pos = 0

    def peek(self) -> Token | None:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def eat(self) -> Token:
        tok = self.toks[self.pos]
        self.pos += 1
        return tok

    def parse_expr(self) -> dict:
        left = self.parse_and()
        operands = [left]
        while self.peek() and self.peek().kind == TOK_OR:
            self.eat()
            operands.append(self.parse_and())
        if len(operands) == 1:
            return operands[0]
        return {"op": "or", "operands": operands}

    def parse_and(self) -> dict:
        left = self.parse_atom()
        operands = [left]
        while self.peek() and self.peek().kind == TOK_AND:
            self.eat()
            operands.append(self.parse_atom())
        if len(operands) == 1:
            return operands[0]
        return {"op": "and", "operands": operands}

    def parse_atom(self) -> dict:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of input")
        if tok.kind == TOK_LPAREN:
            self.eat()
            inner = self.parse_expr()
            if not self.peek() or self.peek().kind != TOK_RPAREN:
                raise ValueError("Expected ')'")
            self.eat()
            return inner
        if tok.kind == TOK_ATOM:
            self.eat()
            return tok.value
        raise ValueError(f"Unexpected token {tok.kind} at position {self.pos}")


def parse_prereq(text: str | None) -> dict | None:
    """Return a boolean expression tree, or None if there is no prereq."""
    if text is None:
        return None
    stripped = text.strip()
    if stripped in EMPTY_MARKERS:
        return None
    # A few rows use "__ <free-text note>" to mean "no prereq, just a note".
    if stripped.startswith("__"):
        return None
    tokens = tokenize(stripped)
    if not tokens:
        return None
    parser = _Parser(tokens)
    tree = parser.parse_expr()
    if parser.pos != len(tokens):
        leftover = tokens[parser.pos:]
        raise ValueError(f"Trailing tokens left unparsed: {leftover}")
    return tree


def flatten_courses(expr: dict | None) -> list[str]:
    """Return every course code appearing in an expression tree."""
    if expr is None:
        return []
    if "course" in expr:
        return [expr["course"]]
    out: list[str] = []
    for op in expr.get("operands", []):
        out.extend(flatten_courses(op))
    # dedup, preserve order
    seen: set[str] = set()
    dedup: list[str] = []
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def expr_satisfied(expr: dict | None, completed: set[str]) -> bool:
    """Evaluate the expression against a set of completed course codes."""
    if expr is None:
        return True
    if "course" in expr:
        return expr["course"] in completed
    op = expr["op"]
    operands = expr["operands"]
    if op == "and":
        return all(expr_satisfied(o, completed) for o in operands)
    if op == "or":
        return any(expr_satisfied(o, completed) for o in operands)
    raise ValueError(f"Unknown op {op!r}")


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    samples = [
        "CS 204 - Undergraduate - Min Grade D",
        "(MATH 204 - Undergraduate - Min Grade D and CS 300 - Undergraduate - Min Grade D)",
        "(MATH 201 - Undergraduate - Min Grade D or MATH 212 - Undergraduate - Min Grade D) and MATH 203 - Undergraduate - Min Grade D",
        "ENS 207 - Undergraduate - Min Grade D or (ME 307 - Undergraduate - Min Grade D and ME 309 - Undergraduate - Min Grade D) or (ME 307 - Undergraduate - Min Grade D and ENS 202 - Undergraduate - Min Grade D) or (ME 309 - Undergraduate - Min Grade D and ENS 202 - Undergraduate - Min Grade D)",
        "SPS 102 - Undergraduate - Min Grade D (can be taken concurrently) and PSY 201 - Undergraduate - Min Grade D",
        "__",
        "",
    ]
    import json as _json
    for s in samples:
        print(s)
        print("  ->", _json.dumps(parse_prereq(s)))
