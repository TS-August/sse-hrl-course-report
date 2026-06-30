import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_success_curve(path):
    xs = []
    ys = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["metric"] == "Test_Success_Rate":
                xs.append(float(row["total_timesteps"]))
                ys.append(float(row["value"]))
    if not xs:
        raise ValueError(f"No Test_Success_Rate rows found in {path}")
    return np.array(xs), np.array(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="exp/AntMaze")
    parser.add_argument("--pattern", default="umaze_1m_seed*")
    parser.add_argument("--output", default="figures/umaze_success_rate.png")
    args = parser.parse_args()

    root = Path(args.root)
    runs = sorted(root.glob(args.pattern))
    curves = []
    for run in runs:
        metrics = run / "metrics.csv"
        if metrics.exists():
            curves.append(read_success_curve(metrics))

    if not curves:
        raise SystemExit(f"No metrics.csv files matched {root / args.pattern}")

    min_last_step = min(x[-1] for x, _ in curves)
    grid = np.linspace(0, min_last_step, 200)
    values = []
    for x, y in curves:
        values.append(np.interp(grid, x, y))

    values = np.vstack(values)
    mean = values.mean(axis=0)
    std = values.std(axis=0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3.2, 2.4), dpi=180)
    plt.plot(grid / 1e6, mean, color="#2ca02c", linewidth=2.0, label="SSE")
    plt.fill_between(grid / 1e6, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                     color="#2ca02c", alpha=0.18, linewidth=0)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel("Timestep")
    plt.ylabel("Success Rate")
    plt.title("(a) U-maze", fontsize=9)
    plt.grid(True, linewidth=0.5, alpha=0.7)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
