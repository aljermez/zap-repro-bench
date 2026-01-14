import csv
import os
from datetime import datetime

from bench_utils import machine_info, time_it
import numpy as np


def workload():
    x = np.random.rand(2_000_000).astype(np.float64)
    y = np.sin(x) + np.cos(x)
    return float(np.sum(y))


def append_csv(path, row, fieldnames):
    file_exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    info = machine_info()
    timing = time_it(workload, repeats=20, warmup=3)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": info["platform"],
        "python": info["python"],
        "processor": info["processor"],
        "repeats": timing["repeats"],
        "warmup": timing["warmup"],
        "min_s": timing["min_s"],
        "median_s": timing["median_s"],
        "max_s": timing["max_s"],
    }

    csv_path = "results/summary.csv"
    fieldnames = list(row.keys())
    append_csv(csv_path, row, fieldnames)

    print("Appended:", csv_path)
    print("Median seconds:", timing["median_s"])
