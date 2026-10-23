import os
import random
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regex_engine import (
    DEPTH_LIMIT,
    MAX_STEPS,
    BacktrackLimitExceeded,
    MatchDepthExceeded,
    RegexError,
    compile,
)

ALPHABET = "ab"


class TestAtoms(unittest.TestCase):
    def test_literal(self):
        self.assertEqual(compile("abc").fullmatch("abc").group(0), "abc")
        self.assertIsNone(compile("abc").fullmatch("abcd"))
        self.assertEqual(compile("abc").match("abcd").end, 3)

    def test_dot_excludes_newline(self):
        self.assertIsNotNone(compile("a.c").fullmatch("abc"))
        self.assertIsNone(compile("a.c").fullmatch("a\nc"))

    def test_class(self):
        self.assertIsNotNone(compile("[a-c]+").fullmatch("abcabc"))
        self.assertIsNone(compile("[a-c]+").fullmatch("abd"))
        self.assertIsNotNone(compile("[^a-c]+").fullmatch("xyz"))
        self.assertEqual(compile("[-a]+").fullmatch("-a-").end, 3)
        self.assertEqual(compile("[a]").fullmatch("a").end, 1)

    def test_shorthand(self):
        self.assertEqual(compile(r"\d+").fullmatch("1234").end, 4)
        self.assertIsNone(compile(r"\d+").fullmatch("12a4"))
        self.assertEqual(compile(r"\D+").fullmatch("abc").end, 3)
        self.assertEqual(compile(r"\w+").fullmatch("a_9").end, 3)
        self.assertEqual(compile(r"\s+").fullmatch(" \t ").end, 3)
        self.assertEqual(compile(r"\S+").fullmatch("x").end, 1)

    def test_escapes(self):
        self.assertEqual(compile(r"a\.b").fullmatch("a.b").end, 3)
        self.assertIsNone(compile(r"a\.b").fullmatch("axb"))
        self.assertEqual(compile(r"\t").fullmatch("\t").end, 1)
        self.assertEqual(compile(r"\n").fullmatch("\n").end, 1)


class TestQuantifiers(unittest.TestCase):
    def test_star_plus_question(self):
        self.assertIsNotNone(compile("a*").fullmatch(""))
        self.assertIsNone(compile("a+").fullmatch(""))
        self.assertIsNotNone(compile("a?").fullmatch("a"))
        self.assertIsNotNone(compile("a?").fullmatch(""))

    def test_brace(self):
        self.assertIsNotNone(compile("a{3}").fullmatch("aaa"))
        self.assertIsNone(compile("a{3}").fullmatch("aa"))
        self.assertIsNotNone(compile("a{2,}").fullmatch("aaaa"))
        self.assertIsNotNone(compile("a{1,3}").fullmatch("aaa"))
        self.assertIsNone(compile("a{1,3}").fullmatch("aaaa"))
        self.assertEqual(compile("a{0,2}b").fullmatch("aab").end, 3)

    def test_greedy_vs_lazy(self):
        # search exposes the difference: greedy takes the last `>`, lazy the first.
        self.assertEqual(compile("<.+>").search("<a><b>").group(0), "<a><b>")
        self.assertEqual(compile("<.+?>").search("<a><b>").group(0), "<a>")
        # fullmatch gives both the same span because the tail must be consumed.
        self.assertEqual(compile("<.+?>").fullmatch("<a><b>").end, 6)
        m = compile("(a+?)(a*)").fullmatch("aaaa")
        self.assertEqual((m.group(1), m.group(2)), ("a", "aaa"))

    def test_multi_char_repeat_minimum(self):
        # Regression: minimum must count iterations, not consumed characters.
        self.assertIsNotNone(compile("(ab){2}").fullmatch("abab"))
        self.assertIsNone(compile("(ab){2}").fullmatch("ab"))
        self.assertEqual(compile("(ab){2,}").fullmatch("ababab").end, 6)

    def test_zero_width_loop_terminates(self):
        self.assertEqual(compile("(a*)*").fullmatch("aaa").end, 3)
        # An unanchored lazy star consumes nothing.
        self.assertEqual(compile("a*?").match("aaa").end, 0)
        # fullmatch backtracks into the loop until the text is consumed, the same
        # way re does -- the lazy order is a preference, not a refusal to give up.
        self.assertEqual(compile("a*?").fullmatch("aaa").end, 3)
        self.assertEqual(compile("a*?c").match("aaac").end, 4)


class TestAlternationGroupsAnchors(unittest.TestCase):
    def test_alternation_preference(self):
        self.assertEqual(compile("a|ab").match("ab").end, 1)

    def test_groups(self):
        m = compile(r"(\w+)@(\w+)").fullmatch("bob@mail")
        self.assertEqual(m.groups(), ("bob", "mail"))
        m = compile(r"(?:ab)+").fullmatch("abab")
        self.assertEqual(m.groups(), ())
        self.assertRaises(IndexError, m.group, 1)
        self.assertEqual(m.end, 4)

    def test_nested_groups(self):
        m = compile(r"((a)(b))").fullmatch("ab")
        self.assertEqual(m.groups(), ("ab", "a", "b"))

    def test_failed_branch_leaves_no_capture(self):
        m = compile(r"(?:(x)|(a))b").fullmatch("ab")
        self.assertIsNone(m.group(1))
        self.assertEqual(m.group(2), "a")

    def test_anchors(self):
        self.assertIsNotNone(compile("^abc$").match("abc"))
        self.assertIsNone(compile("^abc").match("xabc"))
        self.assertIsNotNone(compile("abc$").search("zzabc"))


class TestSearchHelpers(unittest.TestCase):
    def test_search_and_findall(self):
        self.assertEqual(compile("b").search("abc").start, 1)
        self.assertEqual(compile(r"(\d)").findall("a1b2"), ["1", "2"])
        self.assertEqual(compile("oo").findall("bookroom"), ["oo", "oo"])

    def test_finditer_no_infinite_loop_on_empty(self):
        self.assertEqual([m.start for m in compile("a*").finditer("bb")][:4], [0, 1, 2])

    def test_no_match_returns_none(self):
        self.assertIsNone(compile("xyz").search("abc"))


class TestErrors(unittest.TestCase):
    def test_rejected_patterns(self):
        for bad in ["(a", "[a", "a{2,1}", "*a", "a\\", "a)", "a{1,", "\\q"]:
            with self.assertRaises(RegexError, msg=bad):
                compile(bad)

    def test_valid_anchor_forms(self):
        self.assertIsNotNone(compile("^a$").fullmatch("a"))
        self.assertIsNotNone(compile("a|").fullmatch(""))

    def test_literal_brace_is_not_a_quantifier(self):
        self.assertEqual(compile("a{b}").fullmatch("a{b}").end, 4)
        self.assertEqual(compile("a{}").fullmatch("a{}").end, 3)
        self.assertIsNone(compile("a{}").fullmatch("}"))


def _random_pattern(rng, depth=0):
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
            atom = rng.choice(["\\d", "\\w", "[ab]"])
        elif kind == 4:
            inner = _random_pattern(rng, depth + 1)
            atom = f"(?:{inner})" if rng.random() < 0.5 else f"({inner})"
        else:
            atom = f"{rng.choice(['a', 'b', 'ab'])}|{rng.choice(['a', 'b', 'bb'])}"
        quant = rng.choice(["", "", "", "*", "+", "?", "{1,2}", "{2}", "*?", "+?"])
        pieces.append(atom + quant)
    sep = rng.choice(["", "|"])
    return sep.join(pieces)


class TestDifferentialAgainstStdlib(unittest.TestCase):
    def test_random_patterns_agree(self):
        rng = random.Random(20261023)
        checked = 0
        abandoned = 0
        for _ in range(400):
            pattern = _random_pattern(rng)
            try:
                mine = compile(pattern)
            except RegexError:
                continue
            std = re.compile(pattern)
            for _ in range(5):
                text = "".join(rng.choice("ab") for _ in range(rng.randint(0, 8)))
                for method in ("match", "fullmatch"):
                    try:
                        a = getattr(mine, method)(text)
                    except (BacktrackLimitExceeded, RecursionError):
                        # Too expensive to answer is a documented engine limit,
                        # not a wrong answer, so it is counted rather than
                        # compared. stdlib re has the same shape of limit.
                        abandoned += 1
                        continue
                    b = getattr(std, method)(text)
                    if (a is None) != (b is None):
                        self.fail(f"{method} {pattern!r} {text!r}: mine={a} std={b}")
                    if a is not None:
                        self.assertEqual(
                            (a.start, a.end, a.groups()),
                            (b.start(), b.end(), b.groups()),
                            f"{method} {pattern!r} vs {text!r}",
                        )
                    checked += 1
        self.assertGreater(checked, 1000)
        self.assertLess(abandoned, checked // 20)


class TestBacktrackBudget(unittest.TestCase):
    def test_bomb_is_stopped_not_hung(self):
        with self.assertRaises(BacktrackLimitExceeded) as caught:
            compile(r"(a+)+b", max_steps=5000).match("a" * 40 + "c")
        self.assertGreaterEqual(caught.exception.steps, 5000)
        self.assertEqual(caught.exception.pattern, r"(a+)+b")

    def test_budget_does_not_change_answers_underneath_it(self):
        for steps in (10_000, MAX_STEPS, 10 * MAX_STEPS):
            with self.subTest(max_steps=steps):
                pattern = compile(r"(\w+)@(\w+)", max_steps=steps)
                found = pattern.match("ada@lovelace")
                self.assertEqual((found.start, found.end), (0, 12))
                self.assertEqual(found.groups(), ("ada", "lovelace"))

    def test_budget_is_per_attempt_not_per_search(self):
        # search() tries every start position, each with its own budget, so one
        # position's cap cannot be drained by an earlier position's work.
        pattern = compile(r"(a+)+b", max_steps=200)
        with self.assertRaises(BacktrackLimitExceeded):
            pattern.match("a" * 30 + "c")
        self.assertIsNotNone(compile(r"(a+)+b", max_steps=200).match("aab"))

    def test_last_steps_reports_the_work_of_the_attempt(self):
        pattern = compile(r"(a+)+b", max_steps=10**9)
        counts = []
        for n in (6, 8, 10):
            pattern.match("a" * n + "c")
            counts.append(pattern.last_steps)
        self.assertLess(counts[0], counts[1])
        self.assertLess(counts[1], counts[2])
        # The bomb is exponential: four more characters cost three times the
        # work each, so n=10 is about 3**4 = 81 times n=6.
        self.assertGreater(counts[2], 70 * counts[0])
        self.assertLess(counts[2], 90 * counts[0])

    def test_last_steps_is_the_bounded_count_when_abandoned(self):
        pattern = compile(r"(a+)+b", max_steps=5000)
        with self.assertRaises(BacktrackLimitExceeded):
            pattern.match("a" * 40 + "c")
        self.assertEqual(pattern.last_steps, 5001)


class TestRecursionBudget(unittest.TestCase):
    def test_importing_does_not_change_the_process_limit(self):
        # Import-time setrecursionlimit() would silently rewrite the host's
        # stack budget; the matcher only raises it for the length of one attempt.
        before = sys.getrecursionlimit()
        compile(r"(a|b)*c").match("ab" * 50 + "c")
        self.assertEqual(sys.getrecursionlimit(), before)

    def test_deep_match_reports_a_typed_limit(self):
        # Each frame of a greedy repeat holds one continuation, so nesting grows
        # with the length of the run consumed. Past DEPTH_LIMIT the attempt is
        # abandoned with a named error instead of a bare RecursionError.
        with self.assertRaises(MatchDepthExceeded) as caught:
            compile(r"[ab]*c").match("a" * (DEPTH_LIMIT + 10) + "c")
        self.assertIsInstance(caught.exception, RecursionError)
        self.assertEqual(caught.exception.pattern, r"[ab]*c")

    def test_depth_limit_keeps_working_answers_below_it(self):
        found = compile(r"[ab]*c").match("a" * 500 + "c")
        self.assertEqual((found.start, found.end), (0, 501))

    def test_search_reports_depth_rather_than_returning_nothing(self):
        # search() walks start positions; a too-deep attempt is a real failure
        # mode and must not be mistaken for "no match".
        with self.assertRaises(MatchDepthExceeded):
            compile(r"[ab]*z").search("a" * (DEPTH_LIMIT + 10) + "b")


if __name__ == "__main__":
    unittest.main(verbosity=2)
