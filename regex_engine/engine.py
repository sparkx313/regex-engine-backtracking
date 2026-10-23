"""A small backtracking regular-expression engine, written from scratch.

Supported syntax: literals, ``.``, ``[...]`` with ranges and negation, escapes
(``d D w W s S n r t f v 0`` and punctuation), greedy and lazy quantifiers
``* + ? {n,m}``, alternation ``|``, capturing groups ``(...)``, non-capturing
groups ``(?:...)``, and the anchors ``^`` and ``$``.

Matching is continuation-passing style. Every node is handed a ``cont`` callable
meaning "the rest of the pattern succeeded up to here"; a node returns True as
soon as any path through it lets ``cont`` return True. That single rule gives
backtracking, alternation preference, and greedy-vs-lazy behaviour without an
explicit stack.

Group captures live in one mutable list shared by the whole recursion: a group
writes its start index, and on success the continuation writes the completed
``(start, end)`` tuple before propagating True upward. Failed attempts restore
the previous value, so abandoned branches never leak captures.
"""

from __future__ import annotations

import contextlib
import sys

MAX_REPEAT = 1_000_000

# Backtracking has no polynomial worst case, so an attempt needs a ceiling on
# the number of nodes it may visit. Without it a single pattern like `(a+)+b`
# can occupy the process for hours; with it the engine says "too expensive"
# instead of appearing to hang.
MAX_STEPS = 1_000_000

# The matcher recurses once per continuation frame, so a repeat that swallows a
# long run of text needs a stack that deep. Matching raises the interpreter's
# limit only for the duration of one attempt and restores it afterwards, rather
# than changing it for the whole process at import.
DEPTH_LIMIT = 10_000


class RegexError(Exception):
    pass


class BacktrackLimitExceeded(Exception):
    """An attempt visited more than `max_steps` nodes and was abandoned."""

    def __init__(self, pattern, steps):
        super().__init__(f"{pattern!r}: gave up after {steps} backtracking steps")
        self.pattern = pattern
        self.steps = steps


class MatchDepthExceeded(RecursionError):
    """An attempt nested deeper than `DEPTH_LIMIT` frames and was abandoned.

    Subclasses RecursionError so callers that guard against the interpreter
    limit keep working. Raised instead of leaking a raw RecursionError from deep
    inside the matcher, where the caller cannot tell "too deep" from a bug.
    """

    def __init__(self, pattern, steps):
        super().__init__(
            f"{pattern!r}: needed more than {DEPTH_LIMIT} nested frames "
            f"after {steps} backtracking steps"
        )
        self.pattern = pattern
        self.steps = steps


# ---------------------------------------------------------------- AST nodes


class Char:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class AnyChar:
    __slots__ = ()


class Class:
    __slots__ = ("negated", "items")

    def __init__(self, negated, items):
        self.negated = negated
        self.items = items


class AnchorStart:
    __slots__ = ()


class AnchorEnd:
    __slots__ = ()


class Seq:
    __slots__ = ("parts",)

    def __init__(self, parts):
        self.parts = parts


class Alt:
    __slots__ = ("branches",)

    def __init__(self, branches):
        self.branches = branches


class Repeat:
    __slots__ = ("node", "lo", "hi", "lazy")

    def __init__(self, node, lo, hi, lazy):
        self.node = node
        self.lo = lo
        self.hi = hi
        self.lazy = lazy


class Group:
    __slots__ = ("index", "node")

    def __init__(self, index, node):
        self.index = index  # 0 means non-capturing
        self.node = node


ATOM_TYPES = (Char, AnyChar, Class, AnchorStart, AnchorEnd)


def _test_atom(node, text, i):
    """Return the position after this atom, or None if it does not match here."""
    if isinstance(node, Char):
        return i + 1 if i < len(text) and text[i] == node.value else None
    if isinstance(node, AnyChar):
        return i + 1 if i < len(text) and text[i] != "\n" else None
    if isinstance(node, Class):
        if i >= len(text):
            return None
        c = text[i]
        hit = False
        for item in node.items:
            if isinstance(item, tuple):
                if item[0] <= c <= item[1]:
                    hit = True
                    break
            elif c == item:
                hit = True
                break
        return i + 1 if hit != node.negated else None
    if isinstance(node, AnchorStart):
        return i if i == 0 else None
    if isinstance(node, AnchorEnd):
        return i if i == len(text) else None
    return None  # composite node: caller dispatches on its own type


# -------------------------------------------------------------------- parser

_SIMPLE_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "f": "\f", "v": "\v", "0": "\0"}
_NOT_A_QUANTIFIER = object()


class Parser:
    def __init__(self, pattern):
        self.p = pattern
        self.i = 0
        self.groups = 0

    def peek(self):
        return self.p[self.i] if self.i < len(self.p) else None

    def parse(self):
        body = self.parse_alt()
        if self.i != len(self.p):
            raise RegexError(f"unexpected {self.p[self.i]!r} at index {self.i}")
        return body, self.groups

    def parse_alt(self):
        branches = [self.parse_seq()]
        while self.peek() == "|":
            self.i += 1
            branches.append(self.parse_seq())
        return branches[0] if len(branches) == 1 else Alt(branches)

    def parse_seq(self):
        parts = []
        while True:
            c = self.peek()
            if c is None or c in "|)":
                break
            parts.append(self.parse_quantified())
        return Seq(parts)

    def parse_quantified(self):
        atom = self.parse_atom()
        c = self.peek()
        if c == "*":
            self.i += 1
            lo, hi = 0, MAX_REPEAT
        elif c == "+":
            self.i += 1
            lo, hi = 1, MAX_REPEAT
        elif c == "?":
            self.i += 1
            lo, hi = 0, 1
        elif c == "{":
            bounds = self.parse_brace()
            if bounds is _NOT_A_QUANTIFIER:
                # `{` was not a quantifier, so the atom parsed before it stands.
                return atom
            lo, hi = bounds
        else:
            return atom
        lazy = self.peek() == "?"
        if lazy:
            self.i += 1
        return Repeat(atom, lo, hi, lazy)

    def parse_brace(self):
        start = self.i
        self.i += 1
        lo_digits = self._digits()
        nxt = self.peek()
        if nxt == ",":
            self.i += 1
            hi_digits = self._digits()
            lo = int(lo_digits) if lo_digits else 0
            hi = int(hi_digits) if hi_digits else MAX_REPEAT
        elif nxt == "}":
            if not lo_digits:
                # `{}` is not a quantifier; re treats it as literal text.
                self.i = start
                return _NOT_A_QUANTIFIER
            lo = hi = int(lo_digits)
        else:
            self.i = start
            return _NOT_A_QUANTIFIER
        if self.peek() != "}":
            raise RegexError("unterminated { } quantifier")
        self.i += 1
        if hi < lo:
            raise RegexError(f"quantifier range {lo},{hi} is reversed")
        return lo, hi

    def _digits(self):
        out = ""
        while self.peek() is not None and self.peek().isdigit():
            out += self.p[self.i]
            self.i += 1
        return out

    def parse_atom(self):
        c = self.peek()
        if c is None:
            raise RegexError("pattern ended unexpectedly")
        if c == "(":
            return self.parse_group()
        if c == "[":
            return self.parse_class()
        if c == "\\":
            self.i += 1
            return self.parse_escape()
        if c == ".":
            self.i += 1
            return AnyChar()
        if c == "^":
            self.i += 1
            return AnchorStart()
        if c == "$":
            self.i += 1
            return AnchorEnd()
        if c in "*+?":
            raise RegexError(f"quantifier {c!r} has nothing to repeat")
        self.i += 1
        return Char(c)

    def parse_group(self):
        self.i += 1
        if self.p[self.i : self.i + 2] == "?:":
            self.i += 2
            inner = self.parse_alt()
            self.expect(")")
            return Group(0, inner)
        # Number groups by opening-paren order, matching re's numbering.
        self.groups += 1
        index = self.groups
        inner = self.parse_alt()
        self.expect(")")
        return Group(index, inner)

    def expect(self, ch):
        if self.peek() != ch:
            raise RegexError(f"expected {ch!r} at index {self.i}")
        self.i += 1

    def parse_escape(self):
        c = self.peek()
        if c is None:
            raise RegexError("pattern ends with a dangling backslash")
        self.i += 1
        if c in _SIMPLE_ESCAPES:
            return Char(_SIMPLE_ESCAPES[c])
        if c in "dDwWsS":
            return _SHORTHAND[c]
        if c.isalnum():
            raise RegexError(f"unsupported escape \\{c}")
        return Char(c)

    def parse_class(self):
        self.i += 1
        negated = self.peek() == "^"
        if negated:
            self.i += 1
        items = []
        first = True
        while True:
            c = self.peek()
            if c is None:
                raise RegexError("unterminated [ ] class")
            if c == "]" and not first:
                self.i += 1
                break
            first = False
            if c == "\\":
                self.i += 1
                esc = self.peek()
                if esc is None:
                    raise RegexError("dangling backslash inside class")
                self.i += 1
                lo = _SIMPLE_ESCAPES.get(esc, esc)
            else:
                lo = c
                self.i += 1
            if self.peek() == "-" and self.i + 1 < len(self.p) and self.p[self.i + 1] != "]":
                self.i += 1
                hi = self.p[self.i]
                self.i += 1
                if hi < lo:
                    raise RegexError(f"class range {lo}-{hi} is reversed")
                items.append((lo, hi))
            else:
                items.append(lo)
        return Class(negated, items)


_DIGITS = [Class(False, [("0", "9")])]
_WORD = [Class(False, [("a", "z"), ("A", "Z"), ("0", "9"), ("_", "_")])]
_SPACE = [Class(False, [" ", "\t", "\n", "\r", "\f", "\v"])]

_SHORTHAND = {
    "d": _DIGITS[0],
    "D": Class(True, _DIGITS[0].items),
    "w": _WORD[0],
    "W": Class(True, _WORD[0].items),
    "s": _SPACE[0],
    "S": Class(True, _SPACE[0].items),
}


# ------------------------------------------------------------------- matcher


class _State:
    """Mutable matching state shared by one attempt.

    ``starts`` holds group start offsets and is restored whenever an attempt
    fails, so an abandoned branch cannot leave a capture behind. ``spans`` holds
    completed spans and is maintained through ``writes``: an ordered log of the
    ``(slot, previous_value)`` pairs made since the current group began.

    Two rules make the log correct without a success flag:

    * A group only rolls back when its whole attempt failed, which means
      nothing above it can have succeeded, so no outer span can have been
      written in the meantime to be clobbered by the rollback.
    * The tail of the log is the newest entry, and a newer span always belongs
      to a group nested inside, or after, the one being rolled back -- so
      popping back to the checkpoint undoes exactly this group's writes.
    """

    __slots__ = ("starts", "spans", "writes", "steps", "max_steps", "pattern")

    def __init__(self, ngroups, max_steps=MAX_STEPS, pattern=""):
        self.starts = [None] * ngroups
        self.spans = [None] * ngroups
        self.writes = []
        self.steps = 0
        self.max_steps = max_steps
        self.pattern = pattern


def _match(node, text, i, cont, st):
    st.steps += 1
    if st.steps > st.max_steps:
        raise BacktrackLimitExceeded(st.pattern, st.steps)
    if isinstance(node, ATOM_TYPES):
        nxt = _test_atom(node, text, i)
        return nxt is not None and cont(text, nxt, st)
    if isinstance(node, Seq):
        return _match_seq(node.parts, 0, text, i, cont, st)

    if isinstance(node, Alt):
        for branch in node.branches:
            if _match(branch, text, i, cont, st):
                return True
        return False

    if isinstance(node, Group):
        if node.index == 0:
            return _match(node.node, text, i, cont, st)
        return _match_group(node, text, i, cont, st)

    if isinstance(node, Repeat):
        return _match_repeat(node, text, i, cont, st)

    raise RegexError(f"unhandled node {type(node).__name__}")


def _set_span(st, slot, value):
    st.writes.append((slot, st.spans[slot]))
    st.spans[slot] = value


def _rollback(st, checkpoint):
    while len(st.writes) > checkpoint:
        slot, previous = st.writes.pop()
        st.spans[slot] = previous


def _match_group(node, text, i, cont, st):
    slot = node.index - 1
    checkpoint = len(st.writes)
    saved_start = st.starts[slot]
    st.starts[slot] = i

    def on_group_end(t, j, s):
        _set_span(s, slot, (i, j))
        if cont(t, j, s):
            return True
        _rollback(s, checkpoint)
        return False

    if _match(node.node, text, i, on_group_end, st):
        return True
    _rollback(st, checkpoint)
    st.starts[slot] = saved_start
    return False


def _match_seq(parts, k, text, i, cont, st):
    if k == len(parts):
        return cont(text, i, st)
    return _match(parts[k], text, i, _seq_next(parts, k, cont), st)


def _seq_next(parts, k, cont):
    def on_part_done(text, j, st):
        return _match_seq(parts, k + 1, text, j, cont, st)

    return on_part_done


def _match_repeat(node, text, i, cont, st):
    """Backtracking loop.

    ``advance(j, n, empty)`` means ``n`` iterations have completed and the
    cursor sits at ``j``. The legal moves are one more iteration (when
    ``n < hi``) or the tail (when ``n >= lo``); greedy tries them in that
    order, lazy in reverse.

    A zero-width iteration must stop the loop, otherwise `a*` could spin on
    forever, but the stop is only taken once the minimum is reachable, and it
    is only final after *two* empty passes in a row. The single spare pass is
    what `((?:x?)*?){2}y` needs: the first iteration can match nothing, the
    second then gets to expand its own lazy body over the text. Without the
    spare, `(?:\\w*){2}` style minima also cannot be counted up.
    """

    def tail(j, n):
        return n >= node.lo and cont(text, j, st)

    def advance(j, n, empty):
        def after_iteration(t, new_j, s):
            if new_j == j:
                if n + 1 < node.lo:
                    return advance(new_j, n + 1, False)
                if tail(new_j, n + 1):
                    return True
                if n + 1 >= node.hi or empty:
                    return False
                return advance(new_j, n + 1, True)
            if node.lazy:
                return tail(new_j, n + 1) or advance(new_j, n + 1, False)
            return advance(new_j, n + 1, False) or tail(new_j, n + 1)

        can_iterate = n < node.hi
        if node.lazy:
            return tail(j, n) or (can_iterate and _match(node.node, text, j, after_iteration, st))
        if can_iterate and _match(node.node, text, j, after_iteration, st):
            return True
        return tail(j, n)

    return advance(i, 0, False)


# ---------------------------------------------------------------- public API


@contextlib.contextmanager
def _recursion_headroom():
    """Raise the interpreter limit for one attempt, then put it back.

    Only ever raises it: a host that already allows more depth than DEPTH_LIMIT
    is left exactly as it was, so importing and matching here cannot quietly
    make another library's stack shallower.
    """
    previous = sys.getrecursionlimit()
    if previous < DEPTH_LIMIT:
        sys.setrecursionlimit(DEPTH_LIMIT)
    try:
        yield
    finally:
        if previous < DEPTH_LIMIT:
            sys.setrecursionlimit(previous)


class Match:
    __slots__ = ("start", "end", "string", "_caps", "_ngroups")

    def __init__(self, start, end, string, caps, ngroups):
        self.start = start
        self.end = end
        self.string = string
        self._caps = caps
        self._ngroups = ngroups

    def _span(self, n):
        if n == 0:
            return (self.start, self.end)
        if n < 1 or n > self._ngroups:
            raise IndexError(f"no such group: {n}")
        c = self._caps[n - 1]
        return c if isinstance(c, tuple) else None

    def group(self, n=0):
        sp = self._span(n)
        return None if sp is None else self.string[sp[0] : sp[1]]

    def span(self, n=0):
        return self._span(n)

    def groups(self):
        return tuple(self.group(k) for k in range(1, self._ngroups + 1))

    def __bool__(self):
        return True


class CompiledPattern:
    def __init__(self, pattern, max_steps=MAX_STEPS):
        self.pattern = pattern
        self.max_steps = max_steps
        self.last_steps = 0
        self._node, self._ngroups = Parser(pattern).parse()

    def _at(self, text, pos, fullmatch):
        outcome = {}
        st = _State(self._ngroups, self.max_steps, self.pattern)

        def finish(text, end, s):
            if fullmatch and end != len(text):
                return False
            outcome["end"] = end
            outcome["spans"] = s.spans
            return True

        try:
            with _recursion_headroom():
                hit = _match(self._node, text, pos, finish, st)
        except RecursionError:
            raise MatchDepthExceeded(self.pattern, st.steps) from None
        finally:
            # Nodes visited by this attempt. Wall clock depends on the machine;
            # this is the number that shows the search shape, and it is what
            # `max_steps` caps. For `search()` it is the last position tried.
            self.last_steps = st.steps
        if hit:
            return Match(pos, outcome["end"], text, outcome["spans"], self._ngroups)
        return None

    def match(self, text, pos=0):
        return self._at(text, pos, False)

    def fullmatch(self, text):
        return self._at(text, 0, True)

    def search(self, text):
        for i in range(len(text) + 1):
            m = self._at(text, i, False)
            if m is not None:
                return m
        return None

    def finditer(self, text):
        i = 0
        n = len(text)
        while i <= n:
            m = self._at(text, i, False)
            if m is None:
                i += 1
                continue
            yield m
            i = m.end + 1 if m.end == m.start else m.end

    def findall(self, text):
        out = []
        for m in self.finditer(text):
            out.append(m.group(1) if self._ngroups >= 1 else m.group(0))
        return out


def compile(pattern, max_steps=MAX_STEPS):
    return CompiledPattern(pattern, max_steps)


def match(pattern, text):
    return compile(pattern).match(text)


def fullmatch(pattern, text):
    return compile(pattern).fullmatch(text)


def search(pattern, text):
    return compile(pattern).search(text)
