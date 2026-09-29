"""E7: модель джерела без поділу ресурсу, B = 40 МГц: найкраща середня швидкість залежно від M."""
import numpy as np, json
from multiprocessing import Pool
from model import PARAMS, ga, rates
def job(a):
    M, s = a
    p = dict(PARAMS); p.update(B=40e6)
    U = np.random.default_rng(2000 + s).uniform(0, 100, (200, 2))
    x, f, _ = ga(U, M, p, True, False, rng=np.random.default_rng(s))
    R, load = rates(U, x, p, True, False)
    return dict(M=M, seed=s, mean=float(R.mean()/1e6), f=float(f/1e6), overload=int((load > p["n_max"]).sum()))
if __name__ == "__main__":
    res = Pool().map(job, [(M, s) for M in [1, 2, 3, 5, 10, 20] for s in range(5)])
    json.dump(res, open("E7.json", "w"))
    for M in [1, 2, 3, 5, 10, 20]:
        a = [r for r in res if r["M"] == M]
        print(M, f"mean {np.mean([r['mean'] for r in a]):7.2f}  F {np.mean([r['f'] for r in a]):7.2f} overload {np.mean([r['overload'] for r in a]):.1f}")
