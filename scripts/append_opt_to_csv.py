import csv
import json
import os

CSV_PATH = "results/opt_summary.csv"
JSON_PATH = "results/opt_bench.json"

def append_rows(rows):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)

if __name__ == "__main__":
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    base = {
        "timestamp": data["machine"]["timestamp"],
        "platform": data["machine"]["platform"],
        "python": data["machine"]["python"],
        "processor": data["machine"]["processor"],
        "n": data["problem"]["n"],
        "tol": data["problem"]["tol"],
        "max_iters": data["problem"]["max_iters"],
    }

    rows = []
    for method in ["gd", "momentum", "cg"]:
        d = data[method]
        row = dict(base)
        row.update({
            "method": method,
            "iters": d["iters"],
            "time_s": d["time_s"],
            "final_grad_norm": d["final_grad_norm"],
        })
        rows.append(row)

    append_rows(rows)
    print("Appended", len(rows), "rows to", CSV_PATH)
