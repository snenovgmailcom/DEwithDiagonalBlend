import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import concurrent.futures
from functools import partial
import os

from cec2017.functions import (
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
    f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
    f21, f22, f23, f24, f25, f26, f27, f28, f29, f30
)

from differentialevolution_d_blend import differential_evolution as differential_evolution

# === INITIAL CONDITIONS ===
DIMENSIONS = 100
MAX_ITER = 10000
NUM_RUNS = 51
N_JOBS = 80  # Adjust to your CPU
BOUNDS = (-100, 100)
RESULTS_DIR = "results_benchmark"
FILE_STEM = f"{DIMENSIONS}Dim{NUM_RUNS}Runs"
os.makedirs(RESULTS_DIR, exist_ok=True)

def cec_wrapper(f, x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    return f(x)[0]

FUNCTIONS = {
    f'cec2017_f{i}': partial(cec_wrapper, f)
    for i, f in enumerate(
            [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
             f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
             f21, f22, f23, f24, f25, f26, f27, f28, f29, f30
             ], 1)
}

# Adapter functions for DE using callback to collect best-so-far fitness
def de_solve(func, bounds, maxiter, seed):
    best_so_far = []
    def callback(xk, convergence=None):
        best_so_far.append(func(xk))
    result = differential_evolution(
        func,
        bounds=[bounds]*DIMENSIONS,
        maxiter=maxiter,
        recombination=(0.5,1.0),
        seed=seed,
        p_blend=0,
        tol=0,
        atol=1e-3,
        disp=False,
        workers=1,
        callback=callback
    )
    nit = result.nit if hasattr(result, "nit") else maxiter
    # Only store up to nit
    best_so_far = best_so_far[:nit]
    return type('Result', (), {
        'fun': result.fun,
        'nfev': result.nfev,
        'nit': nit,
        'convergence': best_so_far,
        'sf_history': getattr(result, 'sf_history', None),
        'cr_history': getattr(result, 'cr_history', None),
        'p_blend_history': getattr(result, 'p_blend_history', None)
    })()

def de_d_blended_solve(func, bounds, maxiter, seed):
    best_so_far = []
    def callback(xk, convergence=None):
        best_so_far.append(func(xk))
    result = differential_evolution(
        func,
        bounds=[bounds]*DIMENSIONS,
        maxiter=maxiter,
        recombination=(0.9,1.0),
        seed=seed,
        p_blend=(0.7, 1.0),
        tol=0,
        atol=1e-3,
        disp=False,
        workers=1,
        callback=callback
    )
    nit = result.nit if hasattr(result, "nit") else maxiter
    best_so_far = best_so_far[:nit]
    return type('Result', (), {
        'fun': result.fun,
        'nfev': result.nfev,
        'nit': nit,
        'convergence': best_so_far,
        'sf_history': getattr(result, 'sf_history', None),
        'cr_history': getattr(result, 'cr_history', None),
        'p_blend_history': getattr(result, 'p_blend_history', None)
    })()

ALGORITHMS = {
    'DE': de_solve,
    'DE_d_blended': de_d_blended_solve,
}

def single_run(args):
    solver_func, func, bounds, dim, max_iter, run = args
    t0 = time.process_time()
    result = solver_func(func, bounds, max_iter, seed=42+run)
    cpu_time = time.process_time() - t0
    convergence = np.array(result.convergence)
    nit = result.nit if hasattr(result, "nit") else max_iter
    # Only up to actual nit
    convergence = convergence[:nit]
    best_so_far = np.minimum.accumulate(convergence)
    return (
        result.fun,
        cpu_time,
        result.nfev,
        nit,
        best_so_far,
        result.sf_history,
        result.cr_history,
        result.p_blend_history
    )

def run_benchmark(solver_func, func, bounds, dim, runs, max_iter, n_jobs=None):
    results = {k: [] for k in ['best_values', 'runtimes', 'func_evals', 'nits', 'convergence', 'sf_history', 'cr_history', 'p_blend_history']}
    args_list = [(solver_func, func, bounds, dim, max_iter, run) for run in range(runs)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for fun, runtime, nfev, nit, convergence, sf_hist, cr_hist, p_blend_hist in executor.map(single_run, args_list):
            results['best_values'].append(fun)
            results['runtimes'].append(runtime)
            results['func_evals'].append(nfev)
            results['nits'].append(nit)
            results['convergence'].append(convergence)
            results['sf_history'].append(sf_hist)
            results['cr_history'].append(cr_hist)
            results['p_blend_history'].append(p_blend_hist)
    return results

def plot_both_best_so_far(convergences_dict, nits_dict, label_dict, out_file):
    plt.figure()
    color_map = {'DE': 'blue', 'DE_d_blended': 'orange'}
    for alg, all_convergences in convergences_dict.items():
        nits = nits_dict[alg]
        # Compute median nit (rounded down)
        median_nit = int(np.median(nits))
        # Pad or truncate each run to median_nit
        arr = np.array([
            np.pad(c, (0, max(0, median_nit - len(c))), mode='edge')[:median_nit]
            for c in all_convergences
        ])
        median_curve = np.median(arr, axis=0)
        min_curve = np.min(arr, axis=0)
        max_curve = np.max(arr, axis=0)
        iters = np.arange(1, median_nit + 1)
        color = color_map.get(alg, None)
        plt.plot(iters, median_curve, label=f"{label_dict[alg]} median best-so-far", color=color)
        plt.fill_between(iters, min_curve, max_curve, alpha=0.2, color=color)
    plt.xlabel('Iteration')
    plt.ylabel('Best-so-far Fitness')
    plt.yscale('log')
    plt.title('Best-so-far Fitness (Median, Min-Max Interval)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

if __name__ == "__main__":
    all_results = {fname: {} for fname in FUNCTIONS}
    summary_rows = []
    print(f"Starting CEC2017 benchmarks ({NUM_RUNS} runs, {DIMENSIONS}D, {MAX_ITER} iter, n_jobs={N_JOBS})")

    for fname, func in FUNCTIONS.items():
        print(f"\nBenchmarking {fname}...")
        per_func_convergences = {}
        per_func_nits = {}
        label_dict = {}
        for aname, solver_func in ALGORITHMS.items():
            print(f"  {aname}...")
            res = run_benchmark(
                solver_func, func, BOUNDS, DIMENSIONS, NUM_RUNS, MAX_ITER, N_JOBS
            )
            all_results[fname][aname] = res
            per_func_convergences[aname] = res['convergence']
            per_func_nits[aname] = res['nits']
            label_dict[aname] = aname
            vals = np.array(res['best_values'])
            nits = np.array(res['nits'])
            nfevs = np.array(res['func_evals'])
            runtimes = np.array(res['runtimes'])

            summary_rows.append({
                'Function': fname,
                'Algorithm': aname,
                'Mean Fun': vals.mean(),
                'Median Fun': np.median(vals),
                'Max Fun': vals.max(),
                'Min Fun': vals.min(),
                'Mean Nit': nits.mean(),
                'Median Nit': np.median(nits),
                'Max Nit': nits.max(),
                'Min Nit': nits.min(),
                'Mean NFEV': nfevs.mean(),
                'Median NFEV': np.median(nfevs),
                'Max NFEV': nfevs.max(),
                'Min NFEV': nfevs.min(),
                'Mean Time': runtimes.mean(),
                'Median Time': np.median(runtimes),
                'Max Time': runtimes.max(),
                'Min Time': runtimes.min(),
            })
        # Plot both algorithms on the same figure for this function
        plot_file = os.path.join(RESULTS_DIR, f"{FILE_STEM}_{fname}_both_best_so_far.png")
        plot_both_best_so_far(per_func_convergences, per_func_nits, label_dict, plot_file)
        print(f"    Both best-so-far curves saved to {plot_file}")

    # Save all raw results
    pickle_file = os.path.join(RESULTS_DIR, f"{FILE_STEM}_all_results.pkl")
    with open(pickle_file, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nAll raw results saved to {pickle_file}")

    # Save summary table as CSV
    summary_df = pd.DataFrame(summary_rows)
    csv_file = os.path.join(RESULTS_DIR, f"{FILE_STEM}_summary_results.csv")
    summary_df.to_csv(csv_file, index=False)
    print(f"Summary statistics saved to {csv_file}")

    print("\nBenchmarking complete. Plots and data saved in", RESULTS_DIR)
