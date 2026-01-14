import json
import os
import platform
import time
from datetime import datetime

def machine_info():
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
    }

def time_it(fn, repeats=10, warmup=2):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return {
        "repeats": repeats,
        "warmup": warmup,
        "min_s": times[0],
        "median_s": times[len(times)//2],
        "max_s": times[-1],
        "all_s": times,
    }

def save_result(name, result, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Saved:", path)
