import argparse
import subprocess
import sys

STEPS = [
    ("range_bounds", ["-m", "hesoftmax.experiments.range_bounds", "--windows", "8"]),
    ("complexity", ["-m", "hesoftmax.experiments.complexity", "--spreads", "16.0", "105.0",
                    "1103.5"]),
    ("attack", ["-m", "hesoftmax.attack.indcpad"]),
    ("sweep", ["-m", "hesoftmax.experiments.accuracy_sweep"]),
    ("accuracy", ["-m", "hesoftmax.experiments.gpt2_accuracy", "--windows", "24", "--calib", "8",
                  "--s-meas", "105.0"]),
    ("microbench", ["-m", "hesoftmax.experiments.microbench", "--n", "128", "--spreads", "105.0",
                    "1103.5", "--out", "results/mb_all.json"]),
    ("figures", ["-m", "hesoftmax.experiments.figures"]),
    ("tables", ["-m", "hesoftmax.experiments.make_tables"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    for name, args in STEPS:
        if a.only and name not in a.only:
            continue
        print("=== %s ===" % name, flush=True)
        subprocess.run([sys.executable] + args, check=True)


if __name__ == "__main__":
    main()
