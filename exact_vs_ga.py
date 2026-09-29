"""E5: ГА проти повного перебору на сітці кандидатних точок 10x10 (крок 10 м), M = 2, 3."""
import numpy as np, itertools, json, time
from model import PARAMS, ga, rx_power_w, noise_w, fitness
from run_exp import BASE, users_for

def exhaustive(U, M, p, interf=True):
    g = np.arange(5, 100, 10.0)
    sites = np.array([(x, y) for x in g for y in g])
    Ps = rx_power_w(U, sites, p)                      # (N,100)
    N0 = noise_w(p)
    best, bestc = -np.inf, None
    combos = np.array(list(itertools.combinations(range(len(sites)), M)))
    for k in range(0, len(combos), 20000):
        C = combos[k:k+20000]
        P = Ps[:, C].transpose(1, 0, 2)               # (C,N,M)
        j = P.argmax(-1); S = P.max(-1)
        I = P.sum(-1) - S if interf else 0.0
        R = p["B"] * np.log2(1 + S / (I + N0))
        load = np.stack([(j == m).sum(-1) for m in range(M)], -1)
        R = R / np.take_along_axis(load, j, -1)
        f = R.mean(-1) - p["alpha"] * (R < p["R_min"]).mean(-1) - p["beta"] * (load > p["n_max"]).sum(-1)
        # d_min=10 м: сусідні вузли сітки на відстані 10 м, штраф лише для однакових точок (не виникає)
        i = f.argmax()
        if f[i] > best: best, bestc = f[i], sites[C[i]]
    return best, bestc, len(combos)

if __name__ == "__main__":
    out = []
    for M in [2, 3]:
        for s in range(5):
            p = dict(BASE); U = users_for(s, p)
            t = time.time(); fe, xe, n = exhaustive(U, M, p); te = time.time() - t
            runs = []
            for r in range(10):
                t = time.time()
                xg, fg, hist = ga(U, M, p, True, True, rng=np.random.default_rng(r), grid=10.0)
                runs.append(dict(f=float(fg), t=time.time() - t, evals=40 * 151))
            out.append(dict(M=M, seed=s, f_exact=float(fe), n_exact=n, t_exact=te, ga=runs))
            gaps = [(fe - r["f"]) / abs(fe) * 100 for r in runs]
            print(M, s, f"exact {fe/1e6:.3f} ({n} evals, {te:.1f}s) GA best {max(r['f'] for r in runs)/1e6:.3f} "
                  f"gap mean {np.mean(gaps):.2f}% max {np.max(gaps):.2f}% hit {sum(g < 1e-9 for g in gaps)}/10")
    json.dump(out, open("E5.json", "w"))
