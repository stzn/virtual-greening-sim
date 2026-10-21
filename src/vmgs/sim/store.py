"""実行ディレクトリ（runs/<run_id>/）の入出力。

run.json（設定）/ steps.jsonl（毎時の記録）/ checkpoints/step_NNN.json / summary.json / research_notes.md
"""

import json
from pathlib import Path
from typing import Any

from vmgs.obs.telemetry import StepTelemetry
from vmgs.sim.records import Checkpoint, StepRecord


class RunStore:
    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir
        self.meta_path = run_dir / "run.json"
        self.steps_path = run_dir / "steps.jsonl"
        self.summary_path = run_dir / "summary.json"
        self.notes_path = run_dir / "research_notes.md"
        self.telemetry_path = run_dir / "telemetry.jsonl"
        self.traces_path = run_dir / "traces.jsonl"
        self.checkpoint_dir = run_dir / "checkpoints"

    def exists(self) -> bool:
        return self.meta_path.exists()

    def init(self, meta: dict[str, Any], title: str) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        self.steps_path.write_text("", encoding="utf-8")
        self.telemetry_path.write_text("", encoding="utf-8")
        self.traces_path.write_text("", encoding="utf-8")
        self.notes_path.write_text(f"# 研究ログ: {title}\n\n", encoding="utf-8")

    def read_meta(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.meta_path.read_text(encoding="utf-8"))
        return data

    # --- steps ---
    def append_step(self, rec: StepRecord) -> None:
        with self.steps_path.open("a", encoding="utf-8") as f:
            f.write(rec.model_dump_json() + "\n")

    def read_steps(self) -> list[StepRecord]:
        lines = self.steps_path.read_text(encoding="utf-8").splitlines()
        return [StepRecord.model_validate_json(line) for line in lines if line.strip()]

    def truncate_steps(self, n: int) -> None:
        lines = self.steps_path.read_text(encoding="utf-8").splitlines()[:n]
        self.steps_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

    # --- テレメトリ・トレース（steps.jsonl とは別。再現性を損なわない） ---
    def append_telemetry(self, rec: StepTelemetry) -> None:
        with self.telemetry_path.open("a", encoding="utf-8") as f:
            f.write(rec.model_dump_json() + "\n")

    def read_telemetry(self) -> list[StepTelemetry]:
        if not self.telemetry_path.exists():
            return []
        lines = self.telemetry_path.read_text(encoding="utf-8").splitlines()
        return [StepTelemetry.model_validate_json(line) for line in lines if line.strip()]

    def append_trace(self, rec: dict[str, Any]) -> None:
        with self.traces_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def read_traces(self) -> list[dict[str, Any]]:
        if not self.traces_path.exists():
            return []
        lines = self.traces_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def truncate_side_files(self, n: int) -> None:
        """再開時、チェックポイントより先に書かれた分を捨てて steps と揃える。"""
        for path in (self.telemetry_path, self.traces_path):
            if path.exists():
                lines = path.read_text(encoding="utf-8").splitlines()[:n]
                path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

    # --- checkpoints ---
    def _cp_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"step_{step:03d}.json"

    def save_checkpoint(self, cp: Checkpoint) -> None:
        self._cp_path(cp.step).write_text(cp.model_dump_json(), encoding="utf-8")

    def load_checkpoint(self, step: int) -> Checkpoint:
        return Checkpoint.model_validate_json(self._cp_path(step).read_text(encoding="utf-8"))

    def latest_checkpoint_step(self) -> int:
        steps = [int(p.stem.removeprefix("step_")) for p in self.checkpoint_dir.glob("step_*.json")]
        return max(steps, default=0)

    # --- summary ---
    def write_summary(self, summary: dict[str, Any]) -> None:
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def read_summary(self) -> dict[str, Any] | None:
        if not self.summary_path.exists():
            return None
        data: dict[str, Any] = json.loads(self.summary_path.read_text(encoding="utf-8"))
        return data
