import time
import numpy as np
from bench_utils import machine_info, save_result
from opt_bench import make_problem, conjugate_gradient


def run_cg(Q, b, x0, tol=1e-2, max_iters=2000):
    t0 = time.perf_counter()
    x, hist = conjugate_gradient(Q, b, x0, iters=max_iters, tol=tol)
    t1 = time.perf_counter()
    return x, len(hist), hist[-1], (t1 - t0)


if __name__ == "__main__":
    info = machine_info()

    # "Day 1" problem
    Q1, b1 = make_problem(n=1000, seed=1)

    # "Day 2" problem: small change to b (simulates next day / next instance)
    Q2, b2 = make_problem(n=1000, seed=2)

    tol = 1e-2
    max_iters = 2000

    x_zero = np.zeros(Q1.shape[0])

    # Solve day 1 from zero
    x1, it1, final1, t1 = run_cg(Q1, b1, x_zero, tol=tol, max_iters=max_iters)

    # Solve day 2 from zero (cold start)
    x2_cold, it2c, final2c, t2c = run_cg(Q2, b2, x_zero, tol=tol, max_iters=max_iters)

    # Solve day 2 using day 1 solution as a warm start
    x2_warm, it2w, final2w, t2w = run_cg(Q2, b2, x1, tol=tol, max_iters=max_iters)

    result = {
        "machine": info,
        "settings": {"n": int(Q1.shape[0]), "tol": tol, "max_iters": max_iters},
        "day1": {"iters": it1, "final_res": final1, "time_s": t1},
        "day2_cold": {"iters": it2c, "final_res": final2c, "time_s": t2c},
        "day2_warm": {"iters": it2w, "final_res": final2w, "time_s": t2w},
        "improvement": {
            "iters_saved": it2c - it2w,
            "time_saved_s": t2c - t2w,
        },
    }

    save_result("warm_start_bench", result)
    print("Day2 cold iters/time:", it2c, t2c)
    print("Day2 warm iters/time:", it2w, t2w)
    print("Saved iters/time:", it2c - it2w, t2c - t2w)
