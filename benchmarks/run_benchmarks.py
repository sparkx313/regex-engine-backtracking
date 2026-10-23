"""Measure the engine's correctness coverage and its backtracking behaviour.

Three things are measured, because they are the three things a hand-written
backtracking engine gets wrong:

1. agreement with the standard library on randomly generated patterns
2. the exponential blow-up of nested unbounded quantifiers, which the standard
   library shares (it is also a backtracker) and a Thompson NFA would not have
3. where the recursion budget runs out, which is a cost of continuations being
   real Python frames instead of heap records

The second one is reported as node counts, not just time, because the count is
what the engine's `max_steps` bounds. This implementation enumerates three sub-
calls per added character on `(a+)+b` where stdlib re enumerates two; both are
exponential, and the difference is in the table.
"""

import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regex_engine import DEPTH_LIMIT, BacktrackLimitExceeded, RegexError, compile

ALPHABET = "ab"


def random_pattern(rng, depth=0):
    pieces = []
    for _ in range(rng.randint(1, 3)):
        kind = rng.randint(0, 5 if depth < 2 else 3)
        if kind == 0:
            atom = rng.choice("ab")
        elif kind == 1:
            atom = "."
        elif kind == 2:
            atom = rng.choice(["[ab]", "[^a]", "[a-b]"])
        elif kind == 3:
            atom = rng.choice([r"\d", r"\w", "[ab]"])
        elif kind == 4:
            inner = random_pattern(rng, depth + 1)
            atom = f"(?:{inner})" if rng.random() < 0.5 else f"({inner})"
        else:
            atom = f"{rng.choice(['a', 'b', 'ab'])}|{rng.choice(['a', 'b', 'bb'])}"
        quant = rng.choice(["", "", "", "*", "+", "?", "{1,2}", "{2}", "*?", "+?"])
        pieces.append(atom + quant)
    return rng.choice(["", "|"]).join(pieces)


def agreement(rng, n_patterns, n_texts, timeout_s):
    checked = 0
    diffs = []
    slow = 0
    recursion = 0
    aborted = 0
    start = time.time()
    patterns_used = 0
    for _ in range(n_patterns):
        if time.time() - start > timeout_s:
            break
        patterns_used += 1
        pattern = random_pattern(rng)
        try:
            mine = compile(pattern)
        except RegexError:
            continue
        std = re.compile(pattern)
        for _ in range(n_texts):
            text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 7)))
            for method in ("match", "fullmatch", "search"):
                t0 = time.perf_counter()
                try:
                    a = getattr(mine, method)(text)
                except BacktrackLimitExceeded:
                    # The engine stopped rather than running out of time. The
                    # pattern is still correct, just too expensive to answer,
                    # so it is counted instead of compared.
                    aborted += 1
                    continue
                except RecursionError:
                    # A backtracking engine, ours or re's, gives up on deeply
                    # nested quantifiers rather than running forever.
                    recursion += 1
                    continue
                elapsed = time.perf_counter() - t0
                b = getattr(std, method)(text)
                if elapsed > 0.05:
                    slow += 1
                got = (a.start, a.end, a.groups()) if a else None
                want = (b.start(), b.end(), b.groups()) if b else None
                checked += 1
                if got != want:
                    diffs.append((pattern, text, method, got, want))
    return patterns_used, checked, diffs, slow, recursion, aborted


def catastrophic(budget_s=3.0, max_steps=50_000_000):
    """`(a+)+b` is the canonical backtracking bomb: a failing match costs an
    exponential number of steps in the length of the run of `a`s.

    The step budget is raised well above the default here, because the point is
    to watch the cost curve, not to be cut off by the safety limit. The table
    stops when one row exceeds `budget_s`, since the cost doubles per row and a
    fixed upper bound on n would either stop too early or run for hours.

    `last_steps` is the number that matters: wall clock is this machine's Python
    interpreter, steps are the search itself.
    """
    print("Catastrophic backtracking: pattern (a+)+b against 'a'*n + 'c' (no match)")
    print(f"  step budget raised to {max_steps:,} for this measurement;"
          f" stopping when one call exceeds {budget_s}s")
    print(f"{'n':>4} {'ours (s)':>12} {'steps':>12} {'2**n':>10} {'re (s)':>12} {'ratio':>9}")
    hit_budget = False
    for n in range(8, 64):
        text = "a" * n + "c"
        pattern = compile(r"(a+)+b", max_steps=max_steps)
        t0 = time.perf_counter()
        try:
            pattern.match(text)
            ours = time.perf_counter() - t0
            label = f"{ours:>12.4f}"
        except BacktrackLimitExceeded:
            ours = time.perf_counter() - t0
            label = f"{ours:>11.4f}!"
            hit_budget = True
        steps = pattern.last_steps
        t0 = time.perf_counter()
        re.match(r"(a+)+b", text)
        std = time.perf_counter() - t0
        ratio = ours / std if std else float("inf")
        print(f"{n:>4} {label} {steps:>12,} {2 ** n:>10,} "
              f"{std:>12.4f} {ratio:>8.1f}x")
        if ours > budget_s:
            if hit_budget:
                print("  (!) = hit the step budget before finishing the search")
            break


def doubling_check(max_steps=50_000_000, stop_after=10.0):
    """The honest summary of the table: how much extra input costs one more row.

    Starts where a row is still under a second, because the factor is what the
    measurement is for and it needs at least two rows underneath it.
    """
    print("Growth per added 'a' (same pattern, time for a full failing match)")
    previous_steps = previous_time = None
    for n in range(10, 24):
        text = "a" * n + "c"
        pattern = compile(r"(a+)+b", max_steps=max_steps)
        t0 = time.perf_counter()
        try:
            pattern.match(text)
        except BacktrackLimitExceeded:
            print(f"  n={n:>3} aborted at the step budget")
            break
        ours = time.perf_counter() - t0
        steps = pattern.last_steps
        if previous_steps:
            print(f"  n={n:>3} {ours:>9.4f}s {steps:>12,} steps"
                  f"  x{steps / previous_steps:.2f} steps"
                  f"  x{ours / previous_time:.2f} time vs n-1")
        else:
            print(f"  n={n:>3} {ours:>9.4f}s {steps:>12,} steps")
        previous_steps, previous_time = steps, ours
        if ours > stop_after:
            print("  (stopping here; the next row costs about twice this)")
            break


def deepest_answer(pattern, build, cap=30_000):
    """Longest input this pattern still answers before the frame budget runs out.

    Bisection, because the failure is a cliff: one character past the limit the
    attempt raises instead of returning. `cap` means "never broke" if the whole
    range stays answerable.
    """
    lo, hi = 1, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            compile(pattern).match(build(mid))
            lo = mid
        except RecursionError:
            hi = mid - 1
    return lo


def linear_cases():
    """Growing input on patterns with no nested quantifiers.

    Cost per character stays flat, which is the point of a well-behaved pattern,
    but depth grows with the input because each continuation is a Python frame.
    That is a property of this design rather than a bug: a Thompson NFA or a
    heap-allocated backtracker would not hit it.
    """
    print("Well-behaved patterns, growing input")
    cases = [
        (r"[ab]*c", lambda n: "a" * n + "c"),
        (r"(a|b)*c", lambda n: "ab" * (n // 2) + "c"),
        (r"\w+@\w+", lambda n: "a" * n + "@" + "b" * 10),
    ]
    print(f"{'pattern':>14} {'len':>7} {'ours (s)':>10} {'re (s)':>10}")
    for pattern, build in cases:
        for n in (200, 500, 1000, 2000, 4000):
            text = build(n)
            t0 = time.perf_counter()
            try:
                compile(pattern).match(text)
            except RecursionError:
                print(f"{pattern:>14} {len(text):>7} {'too deep':>10} "
                      f"one continuation frame per step")
                break
            ours = time.perf_counter() - t0
            t0 = time.perf_counter()
            re.match(pattern, text)
            std = time.perf_counter() - t0
            print(f"{pattern:>14} {len(text):>7} {ours:>10.4f} {std:>10.4f}")
    print(f"  deepest input each pattern answers, at DEPTH_LIMIT={DEPTH_LIMIT} frames:")
    for pattern, build in cases:
        n = deepest_answer(pattern, build)
        limit = f"{n:,}+" if n == 30_000 else f"{n:,}"
        print(f"    {pattern:>12} {limit:>10} chars")


def reference_growth(first=20, last=24):
    """The same measurement on stdlib re, which is also a backtracker.

    run at n where one call costs tens of milliseconds, because below that the
    clock cannot resolve the difference. This is the number the step counts
    above are compared against: re is the textbook 2 per character.
    """
    print(f"Growth per added 'a' for stdlib re (n={first}..{last})")
    previous = None
    for n in range(first, last + 1):
        text = "a" * n + "c"
        t0 = time.perf_counter()
        re.match(r"(a+)+b", text)
        ours = time.perf_counter() - t0
        factor = f"  x{ours / previous:.2f} vs n-1" if previous else ""
        print(f"  n={n:>3} {ours:>9.4f}s{factor}")
        previous = ours


def main():
    timeout_s = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    rng = random.Random(42)
    print(f"Differential agreement against re (time cap {timeout_s:.0f}s)")
    patterns, checked, diffs, slow, recursion, aborted = agreement(rng, 4000, 4, timeout_s)
    rate = 100.0 * (1 - len(diffs) / checked) if checked else 0.0
    print(f"  patterns tried   : {patterns}")
    print(f"  comparisons      : {checked}")
    print(f"  agreements       : {checked - len(diffs)} ({rate:.3f}%)")
    print(f"  divergences      : {len(diffs)}")
    print(f"  comparisons >50ms: {slow}")
    print(f"  step-budget stops: {aborted}")
    print(f"  recursion limit  : {recursion}")
    for pattern, text, method, got, want in diffs[:5]:
        print(f"    {method} {pattern!r} vs {text!r}: ours={got} re={want}")
    print()
    linear_cases()
    print()
    catastrophic()
    print()
    doubling_check()
    print()
    reference_growth()


if __name__ == "__main__":
    main()
