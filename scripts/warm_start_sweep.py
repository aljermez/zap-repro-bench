import csv
import time
import numpy as np

from opt_bench import make_problem, conjugate_gradient


def run_cg(Q, b, x0, tol=1e-2, max_iters=2000):
    t0 = time.perf_counter()
    x, hist = conjugate_gradient(Q, b, x0, iters=max_iters, tol=tol)
    t1 = time.perf_counter()
    return x, len(hist), float(hist[-1]), (t1 - t0)


if __name__ == "__main__":
    Q, b1 = make_problem(n=1000, seed=1)
    x0 = np.zeros(Q.shape[0])

    tol = 1e-2
    max_iters = 2000

    # Day1 solution
    x1, it1, final1, t1 = run_cg(Q, b1, x0, tol=tol, max_iters=max_iters)

    rng = np.random.default_rng(123)
    delta = rng.standard_normal(b1.shape[0])
    delta /= np.linalg.norm(delta)

    eps_list = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]

    rows = []
    for eps in eps_list:
        b2 = b1 + eps * delta

        _, it_cold, _, t_cold = run_cg(Q, b2, x0, tol=tol, max_iters=max_iters)
        _, it_warm, _, t_warm = run_cg(Q, b2, x1, tol=tol, max_iters=max_iters)

        rows.append({
            "eps": eps,
            "cold_iters": it_cold,
            "warm_iters": it_warm,
            "iters_saved": it_cold - it_warm,
            "cold_time_s": t_cold,
            "warm_time_s": t_warm,
            "time_saved_s": t_cold - t_warm,
        })

    out_path = "results/warm_start_sweep.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("Wrote:", out_path)
    for r in rows:
        print(r)
