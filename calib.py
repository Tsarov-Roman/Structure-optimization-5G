"""E1: калібрування за сценарієм ICAIIT 2025 (без поділу ресурсу, M = 10, N = 200, 100x100 м)
і чутливість результату до параметрів, яких немає у вихідній статті."""
import numpy as np, json, itertools
from multiprocessing import Pool
from model import PARAMS, ga, rates

def job(a):
    B, P, NF, seed = a
    p = dict(PARAMS); p.update(B=B * 1e6, P_bs_dbm=P, nf_db=NF)
    U = np.random.default_rng(2000 + seed).uniform(0, 100, (200, 2))
    x, f, _ = ga(U, 10, p, True, False, rng=np.random.default_rng(seed))
    R, _ = rates(U, x, p, True, False)
    return dict(B=B, P=P, NF=NF, seed=seed, mean=float(R.mean() / 1e6), below5=float((R < 5e6).mean()))

if __name__ == "__main__":
    jobs = list(itertools.product([20, 40, 100], [10, 20, 24], [5, 7, 9], range(3)))
    with Pool() as pool:
        res = pool.map(job, jobs)
    json.dump(res, open("E1.json", "w"))
    for B, P, NF in itertools.product([20, 40, 100], [10, 20, 24], [5, 7, 9]):
        m = [r["mean"] for r in res if (r["B"], r["P"], r["NF"]) == (B, P, NF)]
        print(B, P, NF, f"{np.mean(m):7.2f} ± {np.std(m):5.2f}")
