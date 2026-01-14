import time
import numpy as np
from bench_utils import machine_info, save_result


def make_problem(n=1000, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n // 5))
    Q = A @ A.T + 1e-3 * np.eye(n)  # SPD
    b = rng.standard_normal(n)
    return Q, b


def grad(Q, b, x):
    return Q @ x - b


def power_iteration_lmax(Q, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(Q.shape[0])
    v /= np.linalg.norm(v)
    for _ in range(iters):
        v = Q @ v
        v /= np.linalg.norm(v)
    # Rayleigh quotient ~ largest eigenvalue
    return float(v @ (Q @ v))


def gd(Q, b, x0, lr, iters=2000, tol=1e-2):
    x = x0.copy()
    history = []
    for _ in range(iters):
        g = grad(Q, b, x)
        gn = float(np.linalg.norm(g))
        history.append(gn)
        if gn < tol:
            break
        x -= lr * g
    return x, history


def momentum(Q, b, x0, lr, beta=0.9, iters=2000, tol=1e-2):
    x = x0.copy()
    v = np.zeros_like(x)
    history = []
    for _ in range(iters):
        g = grad(Q, b, x)
        gn = float(np.linalg.norm(g))
        history.append(gn)
        if gn < tol:
            break
        v = beta * v + g
        x -= lr * v
    return x, history


def conjugate_gradient(Q, b, x0, iters=2000, tol=1e-2):
    # Solve Qx=b for SPD Q
    x = x0.copy()
    r = b - Q @ x
    p = r.copy()
    rs_old = float(r @ r)
    history = [np.sqrt(rs_old)]
    if history[-1] < tol:
        return x, history

    for _ in range(iters):
        Qp = Q @ p
        alpha = rs_old / float(p @ Qp)
        x = x + alpha * p
        r = r - alpha * Qp
        rs_new = float(r @ r)
        history.append(np.sqrt(rs_new))
        if history[-1] < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, history


if __name__ == "__main__":
    Q, b = make_problem(n=1000, seed=1)
    x0 = np.zeros(Q.shape[0])

    info = machine_info()

    # Pick a stable GD step size: lr ≈ 1 / L
    L = power_iteration_lmax(Q, iters=20, seed=0)
    lr = 1.0 / L

    tol = 1e-2
    max_iters = 2000

    t0 = time.perf_counter()
    _, h_gd = gd(Q, b, x0, lr=lr, iters=max_iters, tol=tol)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    _, h_m = momentum(Q, b, x0, lr=lr, beta=0.9, iters=max_iters, tol=tol)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    _, h_cg = conjugate_gradient(Q, b, x0, iters=max_iters, tol=tol)
    t5 = time.perf_counter()

    result = {
        "machine": info,
        "problem": {"n": int(Q.shape[0]), "tol": tol, "max_iters": max_iters, "lr": lr, "L_est": L},
        "gd": {"iters": len(h_gd), "final_grad_norm": h_gd[-1], "time_s": t1 - t0},
        "momentum": {"iters": len(h_m), "final_grad_norm": h_m[-1], "time_s": t3 - t2},
        "cg": {"iters": len(h_cg), "final_grad_norm": h_cg[-1], "time_s": t5 - t4},
    }

    save_result("opt_bench", result)
    print("GD iters:", len(h_gd), "time:", t1 - t0, "final:", h_gd[-1])
    print("Momentum iters:", len(h_m), "time:", t3 - t2, "final:", h_m[-1])
    print("CG iters:", len(h_cg), "time:", t5 - t4, "final:", h_cg[-1])
