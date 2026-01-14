\# zap-repro-bench



Reproducible CPU benchmarking + small optimization experiments in Python.



This repo started as a “make my runs repeatable” setup (env checks + saved results). I’m using it to build a research-style workflow I can apply to the ZAP GPU-accelerated SCOPF paper: baselines, controlled changes, and measurable improvements (including warm-start behavior).



---



\## Results (quick)



\- On the SPD benchmark, CG solved in \*\*27 iterations\*\*; GD and Momentum did not reach the tolerance within the set iteration budget in the same setup.

\- Warm-starting CG helped on perturbed problems (example: \*\*cold 27 iters vs warm 19 iters\*\*).

\- Sweep results are logged in `results/warm\_start\_sweep.csv`.



!\[Warm-start sweep](report/warm\_start\_sweep.png)



---



\## What this repo does (right now)



\- Checks environment / versions so runs are comparable

\- Times CPU workloads consistently (warmup + repeats)

\- Saves results to `results/` as JSON and CSV

\- Runs a small SPD quadratic benchmark comparing:

&nbsp; - Gradient Descent (GD)

&nbsp; - Momentum

&nbsp; - Conjugate Gradient (CG)

\- Tests warm-starting (cold vs warm starts) and sweeps perturbation size



---



\## Setup (Windows)



\### 1) Clone

```bash

git clone https://github.com/aljermez/zap-repro-bench.git

cd zap-repro-bench



