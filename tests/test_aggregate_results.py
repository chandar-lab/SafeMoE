import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _write_judgement(path, category, safety, info):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "summary": {"safety_percentage": safety, "informativeness": info},
                "results": [{"category": category}],
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_results_rejects_incomplete_category_sets(tmp_path):
    run_0 = tmp_path / "seed0"
    run_1 = tmp_path / "seed1"
    _write_judgement(run_0 / "a.json", "A", 10, 1)
    _write_judgement(run_0 / "b.json", "B", 20, 2)
    _write_judgement(run_1 / "a.json", "A", 30, 3)

    result = subprocess.run(
        [sys.executable, "-m", "scripts.aggregate_results", "--runs", str(run_0), str(run_1)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "different category set" in result.stderr
    assert "'B'" in result.stderr


def test_aggregate_results_averages_complete_category_sets_by_run(tmp_path):
    run_0 = tmp_path / "seed0"
    run_1 = tmp_path / "seed1"
    _write_judgement(run_0 / "a.json", "A", 10, 1)
    _write_judgement(run_0 / "b.json", "B", 20, 2)
    _write_judgement(run_1 / "a.json", "A", 30, 3)
    _write_judgement(run_1 / "b.json", "B", 40, 4)

    result = subprocess.run(
        [sys.executable, "-m", "scripts.aggregate_results", "--runs", str(run_0), str(run_1)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "average | 2 | 25.0 | 14.1 | 2.50 | 1.41" in result.stdout
