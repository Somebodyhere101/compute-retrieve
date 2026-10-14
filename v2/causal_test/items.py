"""Stimuli for paper 4 (PREREGISTRATION.md, "Stimuli"). Fixed seeds; every prompt uses the
CR gradient's format "What is {a} * {b}?"."""

import random

FMT = "What is {a} {op} {b}?"

# The CR paper's 24 difficulty-gradient items (experiments/fig3_gradient.py), exploratory.
GRADIENT = [
    (1, "+", 1), (2, "+", 2), (5, "+", 5), (7, "*", 8), (9, "*", 9), (6, "*", 7),
    (12, "*", 12), (15, "*", 15), (11, "*", 11), (13, "*", 14), (17, "*", 6), (19, "*", 8),
    (23, "*", 17), (29, "*", 13), (37, "*", 11), (47, "*", 38), (63, "*", 29), (87, "*", 43),
    (347, "+", 289), (891, "-", 456), (123, "*", 45), (789, "+", 654), (1847, "*", 3), (2468, "+", 1357),
]


def familiar(a: int, b: int) -> bool:
    return a == b or a % 10 == 0 or b % 10 == 0 or a in (11, 25, 75) or b in (11, 25, 75)


def novel(a: int, b: int) -> bool:
    return a != b and all(x % 5 != 0 and x != 11 for x in (a, b))


def item(a, b, s, sub, op="*"):
    ans = {"*": a * b, "+": a + b, "-": a - b}[op]
    return dict(id=f"{s}:{a}{op}{b}", set=s, sub=sub, a=a, b=b, op=op, answer=ans,
                prompt=FMT.format(a=a, op=op, b=b), prefill=f"{a} {op} {b} = ")


def build() -> dict[str, list[dict]]:
    pairs = [(a, b) for a in range(11, 100) for b in range(a, 100)]
    rng = random.Random(4)
    m_fam = rng.sample([p for p in pairs if familiar(*p)], 60)
    m_nov = rng.sample([p for p in pairs if novel(*p)], 180)
    used = set(m_fam) | set(m_nov)
    m_pilot = random.Random(40).sample([p for p in pairs if p not in used], 60)
    return {
        "main": [item(a, b, "F", "fact") for a in range(2, 10) for b in range(a, 10)]
        + [item(a, b, "M", "familiar") for a, b in m_fam]
        + [item(a, b, "M", "novel") for a, b in m_nov],
        "pilot": [item(a, b, "F", "fact") for a in range(2, 10) for b in range(2, a)]
        + [item(a, b, "M", "familiar" if familiar(a, b) else "novel" if novel(a, b) else "other")
           for a, b in m_pilot],
        "gradient": [item(a, b, "G", "gradient", op) for a, op, b in GRADIENT],
    }


if __name__ == "__main__":
    S = build()
    for k, v in S.items():
        print(k, len(v), {s: sum(x["sub"] == s for x in v) for s in sorted({x["sub"] for x in v})})
    assert not {x["id"] for x in S["main"]} & {x["id"] for x in S["pilot"]}
