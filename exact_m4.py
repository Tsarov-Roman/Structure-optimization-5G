"""E9: точний контроль для дискретної постановки, M = 2, 3, 4, сітка 10x10 (крок 10 м), базовий сценарій.
Для кожної реалізації користувачів (ті самі, що в E2) повним перебором знаходяться:
  - мінімум частки непокритих eta_1 (критерій покриття, повний опис);
  - максимум штрафної функції F_1 (4).
Результати порівнюються з розв'язками ГА з E2 (неперервна постановка)."""
import numpy as np, itertools, json, time, sys
from multiprocessing import Pool
from model import rx_power_w, noise_w
from run_exp import BASE, users_for

def exact(args):
    seed, M = args
    p = dict(BASE); U = users_for(seed, p)
    g = np.arange(5, 100, 10.0)
    sites = np.array([(x, y) for x in g for y in g])
    Ps = rx_power_w(U, sites, p); N0 = noise_w(p)
    best_eta, best_f = 2.0, -np.inf
    t = time.time()
    for C in chunks(len(sites), M, 20000):
        P = Ps[:, C].transpose(1, 0, 2)
        j = P.argmax(-1); S = P.max(-1); I = P.sum(-1) - S
        R = p["B"] * np.log2(1 + S / (I + N0))
        load = np.stack([(j == m).sum(-1) for m in range(M)], -1)
        R = R / np.take_along_axis(load, j, -1)
        eta = (R < p["R_min"]).mean(-1)
        f = R.mean(-1) - p["alpha"] * eta - p["beta"] * (load > p["n_max"]).sum(-1)
        best_eta = min(best_eta, float(eta.min())); best_f = max(best_f, float(f.max()))
    return dict(seed=seed, M=M, eta_exact=best_eta, f_exact=best_f, t=time.time() - t)

def chunks(n, M, size):
    it = itertools.combinations(range(n), M)
    while True:
        block = list(itertools.islice(it, size))
        if not block: return
        yield np.array(block)

if __name__ == "__main__":
    jobs = [(s, M) for M in (2, 3, 4) for s in range(10)]
    with Pool() as pool:
        res = pool.map(exact, jobs, chunksize=1)
    json.dump(res, open("E9.json", "w"))
    e2 = json.load(open("E2.json")); e2c = json.load(open("E2cov.json"))
    for r in res:
        ga = [x for x in e2 if x["seed"] == r["seed"] and x["M"] == r["M"]][0]
        gc = [x for x in e2c if x["seed"] == r["seed"] and x["M"] == r["M"]][0]
        print(r["M"], r["seed"], f"eta_exact {r['eta_exact']*100:5.1f}%  GA-penalty eta {ga['full']['below']*100:5.1f}%  GA-cov eta {gc['full']['below']*100:5.1f}%  t {r['t']:.0f}s")
