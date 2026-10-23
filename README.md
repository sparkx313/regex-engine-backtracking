# regex-engine-backtracking

A regular-expression engine written from scratch in pure Python: recursive-descent
parser, AST, and a continuation-passing backtracking matcher that reproduces
`re`'s semantics — including the parts people get wrong, like empty iterations and
capture rollback.

No dependencies. Standard library only, Python 3.9+ (measured on CPython 3.11.9,
Windows x64).

```python
from regex_engine import compile

email = compile(r"(\w+)@(\w+)")
m = email.match("ada@lovelace")
m.group(1), m.group(2)          # ('ada', 'lovelace')
m.span(0)                       # (0, 12)
```

## What it supports

| | |
|---|---|
| Atoms | literals, `.` (not newline), `[a-z]`, `[^a]`, ranges and `-`/`]` placement rules |
| Escapes | `\d \D \w \W \s \S \n \r \t \f \v \0`, plus escaped punctuation |
| Quantifiers | `* + ? {n} {n,} {n,m}`, each with a lazy `?` form |
| Groups | capturing `(...)`, non-capturing `(?:...)`, numbered by opening-paren order |
| Anchors | `^` (start of text), `$` (end of text) |
| API | `compile`, `match`, `fullmatch`, `search`, `findall`, `finditer`, and `Match.group/span/groups` |

Deliberately **not** implemented: named groups, backreferences, lookarounds, `\b`,
inline flags, `re.split`/`re.sub`, and multiline mode. That is the whole scope of
this engine, stated here rather than discovered by you at 2 a.m.

## How the matcher works

Every node is handed a `cont` — "the rest of the pattern succeeded up to here" —
and returns `True` as soon as any path through it makes `cont` return `True`:

```python
def _match(node, text, i, cont, st):
    if isinstance(node, ATOM_TYPES):
        nxt = _test_atom(node, text, i)
        return nxt is not None and cont(text, nxt, st)
```

Alternation is a loop over branches. Greedy is `iterate first, tail second`; lazy
is the same two calls in the other order. Backtracking is Python's own call stack,
so there is no explicit stack to maintain and no state machine to unwind.

Two details are where the correctness actually lives:

**Captures.** Group results live in one mutable list shared by the recursion. A
group writes its start, its continuation writes the completed span, and a failed
attempt restores the previous value through an ordered write log
(`regex_engine/engine.py:_match_group`, `_rollback`). An abandoned branch therefore
cannot leak a capture — which is the bug most hand-written engines ship with.

**Empty iterations.** A repeat must stop when an iteration consumes nothing, or
`(?:a?)*` spins forever. But stopping immediately is also wrong: `((?:x?)*?){2}y`
needs the *second* iteration of the outer repeat to get a chance to expand over the
text. So an empty pass is only final after two of them in a row, and the single
spare pass is what `counted zero-width repeats` require
(`regex_engine/engine.py:_match_repeat`). This rule is what the differential fuzzing
below mostly exists to catch.

## Measured behaviour

Run `python benchmarks/run_benchmarks.py 20` to reproduce. Everything under this
heading is output from that script, not a claim.

### Differential agreement with `re`

Random patterns (alternation, nesting, lazy and greedy quantifiers) over a 2-letter
alphabet, compared against `re` for `match`, `fullmatch` and `search` on 0–7
character texts, 20-second cap:

```
patterns tried   : 833
comparisons      : 9989
agreements       : 9988 (99.990%)
divergences      : 1
comparisons >50ms: 8
step-budget stops: 7
```

The one divergence, exactly as reported:

```
fullmatch '[ab]|((.aa{1,2})+?.+?){1,2}|ab|bb*?' vs 'aaababa'
  ours = (0, 7, ('aaababa', None))    re = (0, 7, ('aaababa', 'aaa'))
```

Both engines agree the whole string matches. They disagree about which alternative
inside the counted lazy group ends up owning group 2 — ours abandons a branch that
`re` still has open. It is one case in ten thousand, in the corner where counted
lazy repeats meet zero-width inner iterations, and it is the reason the README says
99.99% instead of 100%.

### Catastrophic backtracking

`(a+)+b` against `a`×n + `c` (no match). Steps are counted by the engine itself, so
this part of the table is machine-independent:

```
 n     ours (s)        steps       2**n       re (s)     ratio
 8       0.1393       42,646        256       0.0002    843.1x
 9       0.5402      127,939        512       0.0001   8855.8x
10       0.8312      383,818      1,024       0.0001  11857.2x
11       2.3314    1,151,455      2,048       0.0001  17231.2x
12       7.6131    3,454,366      4,096       0.0003  21813.9x

Growth per added 'a' (same pattern, time for a full failing match)
  n= 10    0.9484s      383,818 steps
  n= 11    3.0401s    1,151,455 steps  x3.00 steps  x3.21 time vs n-1
  n= 12    7.5433s    3,454,366 steps  x3.00 steps  x2.48 time vs n-1
  n= 13   20.7725s   10,363,099 steps  x3.00 steps  x2.75 time vs n-1

Growth per added 'a' for stdlib re (n=20..24)
  n= 20    0.0478s
  n= 21    0.0968s  x2.03 vs n-1
  n= 22    0.2188s  x2.26 vs n-1
  n= 23    0.4758s  x2.17 vs n-1
  n= 24    0.8612s  x1.81 vs n-1
```

Read the two growth tables together, because that is the honest finding: `re` costs
about **2× per character**, this engine costs **3.00× per character**. Both are
exponential, and that is the whole lesson of `(a+)+b`, but the naive continuation
enumeration here revisits suffixes that a tuned backtracker prunes, so the base is
worse. That is why the `ratio` column grows with `n` (843× at n=8, 21,814× at n=12)
instead of sitting still at some Python-vs-C constant.

### The step budget

Exponential search is a property of the algorithm, so the engine bounds it instead
of pretending it can be avoided. Every attempt counts the nodes it visits and stops
at `MAX_STEPS`:

```
default budget 1,000,000 steps: stopped in 1.752s after 1,000,001 steps
```

`compile(r"(a+)+b", max_steps=...)` sets it per pattern; the exception carries the
count:

```python
from regex_engine import compile, BacktrackLimitExceeded

try:
    compile(r"(a+)+b").match("a" * 40 + "c")
except BacktrackLimitExceeded as exc:
    exc.steps      # 1000001
    exc.pattern    # '(a+)+b'
```

Without it, the same call would run 2^40-ish steps and look like a hang. With it,
"too expensive to answer" is a value the caller can act on. The budget is per
attempt, so `search()` gets a fresh million steps at each start position.

### Recursion depth

Continuations are real Python frames, so nesting depth grows with the length of the
run a greedy repeat swallows. The matcher raises the interpreter limit to
`DEPTH_LIMIT = 10_000` frames *for the duration of one attempt* and restores it
afterwards — importing this module does not change the host process's stack. Past
that, an attempt raises `MatchDepthExceeded` (a subclass of `RecursionError`) rather
than returning a wrong answer:

```
deepest input each pattern answers, at DEPTH_LIMIT=10000 frames:
      [ab]*c      3,327 chars
     (a|b)*c        907 chars
     \w+@\w+      3,315 chars
```

`(a|b)*c` is the interesting row: an alternation inside the repeat costs about three
frames per character instead of one, so it runs out of stack at a third of the
length. This is a property of the design, not a bug — a Thompson NFA or a
heap-allocated backtracker would not have it. Anything over a few thousand
characters of contiguous repeat is outside what this engine is for.

## Tests

```
python -m unittest discover -s tests -p "test_*.py" -t tests

Ran 31 tests in 1.6s

OK
```

31 tests covering atoms, classes and their edge placement rules, counted and lazy
repeats, capture rollback across abandoned branches, group numbering, parser errors
(reversed ranges, unmatched brackets, dangling backslash, `x{2,1}`, a `{` that is not
a quantifier), the step budget, and the depth budget.

Plus a seeded differential fuzz — 400 random patterns × 5 texts × `match`/`fullmatch`
— that fails the suite on the first disagreement in span or captures, and separately
requires fewer than 1 in 20 attempts to be abandoned by the budgets (an engine that
answers nothing is not a passing engine).

## Layout

```
regex_engine/engine.py        parser, AST, matcher, public API  (~600 lines)
tests/test_engine.py          semantics + budgets + fuzz
benchmarks/run_benchmarks.py  everything in the measured section
```

## What I would do next

- Cut the search base from 3× back toward 2× by not re-enumerating the inner repeat
  at every distinct iteration count — the `n·2ⁿ` composition blow-up is the actual
  cost, and pruning it is the difference between this and a tuned backtracker.
- Fix the single known divergence with the counted lazy group.
- Add lookaheads; they are the last common feature the CPS design takes honestly.
