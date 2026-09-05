#!/usr/bin/env python3
"""
Class A cover for the benchmark gate (T1.7). No accelerator, no model, no history file:
every case builds its own history in a temporary directory.

Run with: python3 -m unittest discover -s scripts/tests
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import perf_gate  # noqa: E402
from perf_gate import PASS, RECORD_ONLY, REGRESSION, UNSTABLE  # noqa: E402

TUPLE = {
    "machine": "rtx5090-laptop",
    "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "model": "Llama-3.2-1B-Instruct",
    "quantization": "Q8_0",
    "backend": "cuda",
    "configuration": "standard",
    "tornadovm_version": "5.2.0-jdk21",
}

TOLERANCES = {
    "default": {"tolerance": 0.03, "mode": "gate"},
    "unstable_spread_limit": 0.10,
    "machines": {"github-hosted": {"mode": "record-only"}},
    "tuples": [
        {"match": {"machine": "rtx5090-laptop", "model": "Mistral-7B-Instruct-v0.3"},
         "tolerance": 0.05},
    ],
}


def entry(eval_rate, status=PASS, cache_warm=True, **overrides):
    row = dict(TUPLE)
    row.update({
        "eval_rate": eval_rate,
        "cache_warm": cache_warm,
        "short_commit": "abcdef12",
        "timestamp": "2026-08-27T00:00:00+00:00",
        "gate": {"status": status},
    })
    row.update(overrides)
    return row


def evaluate(samples, history=(), tup=None, cache_warm=True, tolerances=None):
    return perf_gate.evaluate(list(samples), dict(tup or TUPLE), cache_warm, list(history),
                              tolerances if tolerances is not None else TOLERANCES)


class AggregationTest(unittest.TestCase):

    def test_the_metric_is_the_median_not_the_mean(self):
        # One slow run must not drag the aggregate down; that is why the spec says median.
        median, _ = perf_gate.aggregate([80.0, 81.0, 82.0, 83.0, 40.0])
        self.assertEqual(81.0, median)

    def test_spread_is_relative_to_the_median(self):
        _, spread = perf_gate.aggregate([90.0, 100.0, 110.0, 100.0, 100.0])
        self.assertAlmostEqual(0.2, spread)

    def test_a_non_positive_median_is_an_error_not_a_verdict(self):
        with self.assertRaises(perf_gate.GateError):
            perf_gate.aggregate([0.0, 0.0, 0.0])


class VerdictTest(unittest.TestCase):

    def test_no_baseline_records_only(self):
        status, detail = evaluate([80, 81, 80.5, 80.2, 80.8])
        self.assertEqual(RECORD_ONLY, status)
        self.assertIsNone(detail["baseline"])

    def test_within_tolerance_passes(self):
        history = [entry(100.0)]
        status, detail = evaluate([98, 98.5, 98.2, 98.1, 98.3], history)
        self.assertEqual(PASS, status)
        self.assertAlmostEqual(97.0, detail["baseline"]["floor"])

    def test_a_drop_beyond_the_tolerance_fails(self):
        history = [entry(100.0)]
        status, detail = evaluate([96, 96.5, 96.2, 96.1, 96.3], history)
        self.assertEqual(REGRESSION, status)
        self.assertLess(detail["baseline"]["delta"], -0.03)

    def test_exactly_at_the_floor_passes(self):
        history = [entry(100.0)]
        status, _ = evaluate([97.0] * 5, history)
        self.assertEqual(PASS, status)

    def test_an_improvement_passes(self):
        history = [entry(100.0)]
        status, detail = evaluate([120.0] * 5, history)
        self.assertEqual(PASS, status)
        self.assertGreater(detail["baseline"]["delta"], 0)

    def test_a_noisy_run_is_unstable_rather_than_a_regression(self):
        # Wide spread straddling the floor: without the stability check this would fail.
        history = [entry(100.0)]
        status, detail = evaluate([90, 100, 105, 95, 101], history)
        self.assertEqual(UNSTABLE, status)
        self.assertIsNone(detail["baseline"])

    def test_a_cold_run_is_recorded_but_not_gated(self):
        history = [entry(100.0)]
        status, detail = evaluate([50.0] * 5, history, cache_warm=False)
        self.assertEqual(RECORD_ONLY, status)
        self.assertIsNone(detail["baseline"])

    def test_a_wrong_number_of_measured_runs_is_noted_not_silently_accepted(self):
        status, detail = evaluate([80.0, 80.1, 80.2], [entry(80.0)])
        self.assertEqual(PASS, status)
        self.assertTrue(any("3 measured runs" in note for note in detail["notes"]))


class BaselineSelectionTest(unittest.TestCase):

    def test_only_the_same_tuple_is_a_baseline(self):
        for key, other in (("backend", "opencl"), ("quantization", "F16"),
                           ("tornadovm_version", "5.1.0-jdk21"), ("machine", "ci-box"),
                           ("gpu", "RTX 4090"), ("configuration", "prefill-decode"),
                           ("model", "Qwen3-0.6B")):
            with self.subTest(key=key):
                history = [entry(100.0, **{key: other})]
                status, detail = evaluate([50.0] * 5, history)
                self.assertEqual(RECORD_ONLY, status)
                self.assertIsNone(detail["baseline"])

    def test_the_most_recent_passing_entry_wins(self):
        history = [entry(100.0), entry(80.0)]
        _, detail = evaluate([79.0] * 5, history)
        self.assertEqual(80.0, detail["baseline"]["eval_rate"])

    def test_a_failed_entry_is_not_a_baseline(self):
        # A regression must not become the new bar that the next run is measured against.
        history = [entry(100.0), entry(80.0, status=REGRESSION)]
        _, detail = evaluate([98.0] * 5, history)
        self.assertEqual(100.0, detail["baseline"]["eval_rate"])

    def test_a_record_only_entry_is_a_baseline(self):
        history = [entry(90.0, status=RECORD_ONLY)]
        _, detail = evaluate([89.0] * 5, history)
        self.assertEqual(90.0, detail["baseline"]["eval_rate"])

    def test_legacy_rows_without_the_new_fields_are_never_a_baseline(self):
        legacy = {"model": "Llama-3.2-1B-Instruct", "quantization": "Q8_0",
                  "backend": "cuda", "configuration": "standard", "eval_rate": 200.0}
        status, detail = evaluate([50.0] * 5, [legacy])
        self.assertEqual(RECORD_ONLY, status)
        self.assertIsNone(detail["baseline"])

    def test_a_cold_baseline_is_not_compared_against_a_warm_run(self):
        history = [entry(100.0, cache_warm=False)]
        status, detail = evaluate([50.0] * 5, history)
        self.assertEqual(RECORD_ONLY, status)
        self.assertIsNone(detail["baseline"])


class PolicyTest(unittest.TestCase):

    def test_the_default_applies_when_nothing_matches(self):
        policy = perf_gate.resolve_policy(TOLERANCES, dict(TUPLE, machine="unknown-box"))
        self.assertEqual(0.03, policy["tolerance"])
        self.assertEqual("gate", policy["mode"])

    def test_a_machine_can_be_record_only(self):
        tup = dict(TUPLE, machine="github-hosted")
        status, detail = evaluate([10.0] * 5, [entry(100.0, machine="github-hosted")], tup=tup)
        self.assertEqual(RECORD_ONLY, status)
        self.assertIn("record-only", detail["notes"][0])

    def test_a_tuple_rule_beats_the_machine_rule(self):
        tup = dict(TUPLE, model="Mistral-7B-Instruct-v0.3")
        policy = perf_gate.resolve_policy(TOLERANCES, tup)
        self.assertEqual(0.05, policy["tolerance"])
        self.assertTrue(policy["source"].startswith("tuple:"))

    def test_the_more_specific_of_two_matching_tuple_rules_wins(self):
        tolerances = {
            "default": {"tolerance": 0.03, "mode": "gate"},
            "tuples": [
                {"match": {"machine": "rtx5090-laptop"}, "tolerance": 0.04},
                {"match": {"machine": "rtx5090-laptop", "backend": "cuda"}, "tolerance": 0.02},
            ],
        }
        self.assertEqual(0.02, perf_gate.resolve_policy(tolerances, TUPLE)["tolerance"])

    def test_missing_tolerances_fall_back_to_the_documented_defaults(self):
        policy = perf_gate.resolve_policy({}, TUPLE)
        self.assertEqual(perf_gate.DEFAULT_TOLERANCE, policy["tolerance"])
        self.assertEqual(perf_gate.DEFAULT_SPREAD_LIMIT, policy["spread_limit"])

    def test_the_committed_tolerances_file_parses_and_gates_the_pinned_machine(self):
        path = Path(__file__).resolve().parents[1] / "perf-gate-tolerances.json"
        tolerances = perf_gate.load_tolerances(path)
        policy = perf_gate.resolve_policy(tolerances, TUPLE)
        self.assertEqual("gate", policy["mode"])
        self.assertEqual(0.03, policy["tolerance"])


class MetricsDirTest(unittest.TestCase):

    def test_samples_come_from_the_metrics_files_and_ignore_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "rep1.json").write_text(json.dumps({"eval_rate": 80.0}))
            (directory / "rep2.json").write_text(json.dumps({"eval_rate": 81.0}))
            (directory / "rep2.meta.json").write_text(json.dumps({"eval_rate": 999.0}))
            (directory / "broken.json").write_text("{not json")
            (directory / "failed.json").write_text(json.dumps({"eval_rate": None}))
            self.assertEqual([80.0, 81.0], perf_gate.samples_from_metrics_dir(directory))

    def test_a_directory_with_no_measurements_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(perf_gate.GateError):
                perf_gate.samples_from_metrics_dir(tmp)


class PairedTest(unittest.TestCase):
    """Paired A/B: two builds measured in one session, so drift cancels instead of accumulating."""

    def paired(self, baseline, candidate, tup=None, label="HEAD~1"):
        return perf_gate.evaluate_paired(list(baseline), list(candidate),
                                         dict(tup or TUPLE), TOLERANCES, label)

    def test_equal_builds_pass_regardless_of_the_absolute_level(self):
        # Both sides measured on a hot machine: the level is down, the comparison is not.
        status, detail = self.paired([167.0] * 5, [166.9] * 5)
        self.assertEqual(PASS, status)
        self.assertAlmostEqual(0.0, detail["baseline"]["delta"], places=2)

    def test_a_real_slowdown_is_still_caught(self):
        status, _ = self.paired([167.0] * 5, [150.0] * 5)
        self.assertEqual(REGRESSION, status)

    def test_history_is_never_consulted(self):
        # A stored entry that would fail the history gate must not affect a paired verdict.
        history_entry = entry(1000.0)
        self.assertEqual(REGRESSION, evaluate([100.0] * 5, [history_entry])[0])
        self.assertEqual(PASS, self.paired([100.0] * 5, [100.0] * 5)[0])

    def test_instability_on_either_side_blocks_the_verdict(self):
        self.assertEqual(UNSTABLE, self.paired([90, 100, 105, 95, 101], [100.0] * 5)[0])
        self.assertEqual(UNSTABLE, self.paired([100.0] * 5, [90, 100, 105, 95, 101])[0])

    def test_unequal_run_counts_are_noted_because_interleaving_stops_cancelling(self):
        _, detail = self.paired([167.0] * 5, [167.0] * 3)
        self.assertTrue(any("unequal run counts" in note for note in detail["notes"]), detail["notes"])

    def test_the_record_says_which_kind_of_comparison_produced_it(self):
        status, detail = self.paired([167.0] * 5, [167.0] * 5)
        record = perf_gate.build_record(dict(TUPLE), True, status, detail, _Args())
        self.assertEqual("paired", record["gate"]["comparison"])
        self.assertEqual("HEAD~1", record["gate"]["baseline"]["commit"])

    def test_a_history_verdict_is_labelled_as_such(self):
        status, detail = evaluate([98.0] * 5, [entry(100.0)])
        record = perf_gate.build_record(dict(TUPLE), True, status, detail, _Args())
        self.assertEqual("history", record["gate"]["comparison"])


class _Args:
    """The subset of the CLI namespace build_record reads."""
    commit = "0123456789abcdef"
    branch = "refactor/framework-abstractions"
    run_id = ""
    run_number = ""
    run_attempt = "1"
    workflow = "test"
    warmup_runs = 3


class CliTest(unittest.TestCase):

    def run_gate(self, samples, history_path, extra=()):
        argv = ["--samples", ",".join(str(s) for s in samples),
                "--history", str(history_path),
                "--tolerances", str(Path(__file__).resolve().parents[1] / "perf-gate-tolerances.json"),
                "--cache-warm", "true"]
        for key, value in TUPLE.items():
            argv += ["--" + key.replace("_", "-"), value]
        argv += list(extra)
        return perf_gate.main(argv)

    def test_a_first_run_records_and_the_second_gates_against_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "perf-history.jsonl"

            self.assertEqual(0, self.run_gate([100.0] * 5, history, ["--append"]))
            rows = [json.loads(line) for line in history.read_text().splitlines()]
            self.assertEqual(1, len(rows))
            self.assertEqual(RECORD_ONLY, rows[0]["gate"]["status"])
            self.assertEqual(100.0, rows[0]["eval_rate"])

            self.assertEqual(0, self.run_gate([99.0] * 5, history, ["--append"]))
            self.assertEqual(1, self.run_gate([90.0] * 5, history, ["--append"]))

            rows = [json.loads(line) for line in history.read_text().splitlines()]
            self.assertEqual([RECORD_ONLY, PASS, REGRESSION],
                             [r["gate"]["status"] for r in rows])

            # The regression is recorded but must not become the next baseline.
            self.assertEqual(0, self.run_gate([99.0] * 5, history, ["--append"]))
            rows = [json.loads(line) for line in history.read_text().splitlines()]
            self.assertEqual(99.0, rows[-1]["gate"]["baseline"]["eval_rate"])

    def test_an_unstable_run_exits_two_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "perf-history.jsonl"
            self.assertEqual(2, self.run_gate([90, 100, 105, 95, 101], history, ["--append"]))
            self.assertFalse(history.exists())

    def test_a_usage_error_exits_three_rather_than_reporting_a_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "perf-history.jsonl"
            self.assertEqual(3, self.run_gate([100.0] * 5, history,
                                              ["--cache-warm", "sometimes"]))

    def test_paired_mode_runs_end_to_end_without_a_history_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "perf-history.jsonl"
            code = self.run_gate([167.0] * 5, history,
                                 ["--baseline-samples", "166.9,167.1,167.0,166.8,167.2",
                                  "--baseline-label", "HEAD~1"])
            self.assertEqual(0, code)
            self.assertFalse(history.exists())

    def test_the_record_carries_the_tuple_the_samples_and_the_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "perf-history.jsonl"
            self.assertEqual(0, self.run_gate([80.0, 81.0, 80.5, 80.2, 80.8], history,
                                              ["--append", "--commit", "0123456789abcdef",
                                               "--branch", "refactor/framework-abstractions"]))
            row = json.loads(history.read_text().splitlines()[0])
            for key, value in TUPLE.items():
                self.assertEqual(value, row[key])
            self.assertEqual("01234567", row["short_commit"])
            self.assertEqual(5, len(row["eval_rate_samples"]))
            self.assertEqual(3, row["warmup_runs"])
            self.assertIn("status", row["gate"])


if __name__ == "__main__":
    unittest.main()
