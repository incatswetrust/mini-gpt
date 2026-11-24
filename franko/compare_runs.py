"""Plot val bits per character of the BPE and character runs side by side.

  python franko/compare_runs.py   # -> franko/bpc_compare.png
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {"bpe4000": "BPE, 4000 tokens", "char": "characters, 143 tokens"}


def read_log(name):
    with open(os.path.join(HERE, "out", name, "log.csv")) as f:
        rows = list(csv.DictReader(f))[1:]  # step 0 is ln(vocab), off the scale
    return [int(r["step"]) for r in rows], [float(r["elapsed_s"]) / 60 for r in rows], [float(r["val_bpc"]) for r in rows]


def main():
    fig, (by_step, by_time) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for name, label in RUNS.items():
        steps, minutes, bpc = read_log(name)
        by_step.plot(steps, bpc, marker="o", markersize=3, label=label)
        by_time.plot(minutes, bpc, marker="o", markersize=3, label=label)
    by_step.set(xlabel="step", ylabel="val bits per character", title="per training step")
    by_time.set(xlabel="minutes on M2 Air", title="per wall-clock minute")
    for ax in (by_step, by_time):
        ax.grid(alpha=0.3)
    by_step.legend()
    fig.suptitle("Same GPT (6 layers, 384 dims) on Franko: lower is better")
    fig.tight_layout()
    out = os.path.join(HERE, "bpc_compare.png")
    fig.savefig(out, dpi=120)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
