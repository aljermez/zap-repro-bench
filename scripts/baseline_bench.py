import numpy as np
from bench_utils import machine_info, save_result, time_it

def workload():
    x = np.random.rand(2_000_000).astype(np.float64)
    y = np.sin(x) + np.cos(x)
    return float(np.sum(y))

if __name__ == "__main__":
    info = machine_info()
    timing = time_it(workload, repeats=10, warmup=2)
    result = {"machine": info, "timing": timing}
    save_result("baseline_cpu", result)
