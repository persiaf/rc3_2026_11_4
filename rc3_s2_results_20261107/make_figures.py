"""Run one reproducible script per figure against saved analysis CSVs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=Path(__file__).parent / "analysis")
    args = parser.parse_args()
    started = time.perf_counter()
    report = Path(__file__).parent.resolve()
    analysis = args.analysis.resolve()
    if not analysis.is_relative_to(report):
        raise ValueError("Figure inputs/outputs must remain in the new report directory")
    figure_folder = analysis / "figures"
    receipt = figure_folder / "FIGURE_BUILD.json"
    if receipt.exists():
        raise FileExistsError("Existing figure build receipt is not overwritten")
    for script in sorted(report.glob("gen_fig*.py")):
        subprocess.run([sys.executable, "-B", str(script), "--analysis", str(args.analysis)], check=True)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    outputs = sorted([*figure_folder.glob("*.pdf"), *figure_folder.glob("*.png")])
    if len(outputs) != 18 or any(p.stat().st_size == 0 for p in outputs):
        raise RuntimeError("Expected nine nonempty PDF and nine nonempty PNG files")
    data = dict(schema="rc3-s2-analysis-only-figure-build-v1", runtime_changed=False,
        scientific_optimizer_inference_scene_navigation_teacher_H_calls=0,
        python=sys.version, executable=sys.executable, wall_s=time.perf_counter() - started,
        source_sha256={p.name: sha(p) for p in sorted(report.glob("*.py")) if p.name.startswith(("gen_fig", "plot_s2", "paper_plot_style", "make_figures"))},
        input_csv_sha256={p.name: sha(p) for p in (analysis / "updates.csv", analysis / "group_summary.csv")},
        outputs=[dict(file=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in outputs])
    with receipt.open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(receipt)
