"""The local prover's pipeline for one theorem: the ladder in order, stopping at the first closure,
the run's directory mirroring the winning stage so `local_prover_result` reads it as one run."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import src.local_pipeline as pl


def _spec(run_dir: Path) -> None:
    job = {"run_id": "t-1", "theorem_id": "t", "status": "residual", "file": "F", "declaration": "D"}
    whole = {**job, "status": "bare", "file": "WF", "declaration": "WD"}
    (run_dir / "pipeline.json").write_text(json.dumps({"job": job, "whole_job": whole}))


def _worker(outcomes: dict[str, dict]):
    """A stand-in for `python -m src.local_prover <stage>/job.json`: writes what the stage produced."""

    def run(cmd, **_kw):
        d = Path(cmd[-1]).parent
        got = outcomes.get(d.name, {})
        (d / "status.json").write_text(json.dumps({"state": "done", "holes": {}, "models": {}}))
        (d / "outputs").mkdir(exist_ok=True)
        with open(d / "samples.jsonl", "w") as f:
            for i, (outcome, stop) in enumerate(got.get("samples", [])):
                sid = f"h1-{d.name}-s{i}"
                (d / "outputs" / f"{sid}.txt").write_text("answer")
                f.write(json.dumps({"id": sid, "outcome": outcome, "stop_type": stop, "errors": ["e"] * (i + 1)}) + "\n")
        (d / "fills.json").write_text(json.dumps(got.get("fills", [])))

    return run


class PipelineTest(unittest.TestCase):
    def _run(self, outcomes):
        run_dir = Path(tempfile.mkdtemp())
        _spec(run_dir)
        with mock.patch.object(pl.subprocess, "run", side_effect=_worker(outcomes)) as sub:
            pl.run(run_dir)
        return run_dir, [Path(c.args[0][-1]).parent.name for c in sub.call_args_list]

    def test_the_first_stage_that_closes_ends_the_pipeline_and_becomes_the_run(self):
        fill = [{"hole": 1, "fill": "omega", "helpers": ""}]
        run_dir, ran = self._run({"8b": {"samples": [("verified", "eos")], "fills": fill}})
        self.assertEqual(ran, ["8b"])
        self.assertEqual(json.loads((run_dir / "status.json").read_text())["stage"], "8b")
        self.assertFalse(json.loads((run_dir / "fills.json").read_text())[0]["whole"])

    def test_the_long_stage_runs_only_where_answers_ran_out_of_tokens(self):
        _, ran = self._run({"8b": {"samples": [("wrong_proof", "eos")]}})
        self.assertEqual(ran, ["8b", "whole", "32b"])
        _, ran = self._run({"8b": {"samples": [("truncated", "limit")]}})
        self.assertEqual(ran, ["8b", "8b-long", "whole", "32b"])

    def test_the_32b_revises_the_8bs_best_rejected_proofs(self):
        run_dir, _ = self._run({"8b": {"samples": [("wrong_proof", "eos"), ("wrong_proof", "eos")]}})
        job = json.loads((run_dir / "stages" / "32b" / "job.json").read_text())
        self.assertEqual(job["model"], "goedel-v2-32b")
        self.assertEqual([s["id"] for s in job["seed"]["samples"]], ["h1-8b-s0", "h1-8b-s1"])  # fewest errors first

    def test_a_whole_statement_closure_is_marked_and_its_job_mirrored(self):
        fill = [{"hole": 1, "fill": "simp", "helpers": ""}]
        run_dir, ran = self._run({"8b": {"samples": [("wrong_proof", "eos")]}, "whole": {"fills": fill}})
        self.assertEqual(ran, ["8b", "whole"])
        self.assertTrue(json.loads((run_dir / "fills.json").read_text())[0]["whole"])
        self.assertEqual(json.loads((run_dir / "job.json").read_text())["file"], "WF")

    def test_every_stages_answers_are_counted_in_the_run(self):
        run_dir, _ = self._run({"8b": {"samples": [("wrong_proof", "eos")]}})
        stages = [json.loads(x)["stage"] for x in (run_dir / "samples.jsonl").read_text().splitlines()]
        self.assertEqual(stages, ["8b"])


if __name__ == "__main__":
    unittest.main()
