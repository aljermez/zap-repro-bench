import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":
    in_path = Path("results") / "warm_start_sweep.csv"
    out_dir = Path("report")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "warm_start_sweep.png"

    df = pd.read_csv(in_path)

    # Plot iteration savings vs eps
    plt.figure()
    plt.plot(df["eps"], df["iters_saved"], marker="o")
    plt.xlabel("eps (perturbation size)")
    plt.ylabel("iters_saved (cold - warm)")
    plt.title("Warm-start benefit vs perturbation size")
    plt.grid(True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Saved plot:", out_path)
