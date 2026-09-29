import numpy as np, json, itertools, sys, time
from multiprocessing import Pool
from model import PARAMS, ga, rates, fitness

BASE = dict(PARAMS); BASE.update(B=40e6, R_min=1e6)
MS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
SEEDS = range(10)

def users_for(seed, p):
    return np.random.default_rng(1000 + seed).uniform(0, p["A"], (p["N"], 2))

def metrics(U, X, p, interf):
    R, load = rates(U, X, p, interf, True)
    return dict(mean=float(R.mean()), p5=float(np.percentile(R, 5)), below=float((R < p["R_min"]).mean()), below_t={str(t): float((R < t*1e6).mean()) for t in (0.5, 1, 1.5, 2, 3)})

def job(args):
    tag, pover, M, seed = args
    p = dict(BASE); p.update(pover)
    U = users_for(seed, p)
    xf, _, _ = ga(U, M, p, True, True, rng=np.random.default_rng(seed * 100 + M))
    xr, _, _ = ga(U, M, p, False, True, rng=np.random.default_rng(seed * 100 + M))
    # INR: середнє відношення завада/шум у користувачів для розв'язку повної задачі
    from model import rx_power_w, noise_w
    P = rx_power_w(U, xf, p); S = P.max(-1); I = P.sum(-1) - S
    q = I / noise_w(p)
    inr = float(np.median(q)) if M > 1 else 0.0
    inr90 = float(np.percentile(q, 90)) if M > 1 else 0.0
    inr_mean = float(q.mean()) if M > 1 else 0.0
    P0 = rx_power_w(U, xr, p); S0 = P0.max(-1); q0 = (P0.sum(-1) - S0) / noise_w(p)
    inr0 = float(np.median(q0)) if M > 1 else 0.0
    inr0_90 = float(np.percentile(q0, 90)) if M > 1 else 0.0
    return dict(tag=tag, pover=pover, M=M, seed=seed,
                full=metrics(U, xf, p, True), red=metrics(U, xr, p, False),
                red_true=metrics(U, xr, p, True), inr_med=inr, inr90=inr90, inr_mean=inr_mean, inr0_med=inr0, inr0_90=inr0_90, xf=xf.tolist(), xr=xr.tolist())

if __name__ == "__main__":
    which = sys.argv[1]
    jobs = []
    if which == "E2":
        jobs = [("E2", {}, M, s) for M in MS for s in range(30)]
    elif which == "E2cov":
        jobs = [("E2cov", {"obj": "cov"}, M, s) for M in MS for s in range(30)]
    elif which == "E4":
        for P in [-10.0, 0.0, 10.0, 20.0]:
            for A in [100.0, 300.0]:
                jobs += [("E4", dict(P_bs_dbm=P, A=A), M, s) for M in [2, 5, 10, 20] for s in range(10)]
    elif which == "E6":
        for P in [-10.0, 0.0, 10.0, 20.0]:
            for A in [100.0, 300.0]:
                for pv in [dict(v=2.5), dict(v=3.5), dict(N=100)]:
                    d = dict(P_bs_dbm=P, A=A); d.update(pv)
                    jobs += [("E6", d, M, s) for M in [2, 5, 10, 20] for s in range(5)]
    elif which == "E8":
        for W in [10.0, 20.0, 50.0]:
            for Lw in [2.7, 5.3, 19.0]:
                jobs += [("E8", dict(W=W, Lw=Lw), M, s) for M in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20] for s in range(10)]
    t = time.time()
    with Pool() as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{which}.json", "w"))
    print(which, len(res), f"{time.time()-t:.0f}s")
