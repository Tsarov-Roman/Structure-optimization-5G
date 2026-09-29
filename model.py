"""Відтворення моделі розміщення БС 5G у приміщенні (Siden, Tsarov et al., ICAIIT 2025, формули (1)-(8)).
Параметри, яких немає в статті, задано явно і винесено в PARAMS."""
import numpy as np

PARAMS = dict(
    A=100.0,          # сторона квадратної зони, м
    N=200,            # кількість користувачів
    fc_ghz=3.5,       # несуча, ГГц
    B=100e6,          # смуга, Гц
    P_bs_dbm=20.0,    # потужність БС, дБм
    G_bs_dbi=0.0,     # КП антени БС, дБі
    nf_db=7.0,        # коефіцієнт шуму приймача, дБ
    v=3.0,            # показник втрат, формула (5)
    d0=1.0,           # опорна відстань, м
    h=2.0,            # різниця висот антен, м (d3D)
    R_min=5e6,        # мінімальна швидкість, біт/с
    n_max=40,         # максимальна кількість користувачів на БС
    d_min=10.0,       # мінімальна відстань між БС, м
    alpha=50e6,       # штраф за частку користувачів з R < R_min, біт/с на одиницю частки
    beta=2.0e6,       # штраф за перевантаження БС (за кожну)
    gamma=2.0e6,      # штраф за пару БС ближче d_min
)

def pl0_db(p):
    # вільний простір на 1 м: 32.4 + 20 lg(fc, ГГц), 3GPP TR 38.901 (InH, LOS, d=1 м)
    return 32.4 + 20 * np.log10(p["fc_ghz"])

def noise_w(p):
    return 10 ** ((-174 + 10 * np.log10(p["B"]) + p["nf_db"] - 30) / 10)

def rx_power_w(users, bs, p):
    """users (N,2), bs (...,M,2) -> (...,N,M) прийнята потужність, Вт. Формули (4), (5) без перешкод і завмирань."""
    d2 = ((users[None, :, None, :] - bs[..., None, :, :]) ** 2).sum(-1) if bs.ndim == 3 else \
         ((users[:, None, :] - bs[None, :, :]) ** 2).sum(-1)
    d = np.sqrt(d2 + p["h"] ** 2)
    pl_db = pl0_db(p) + 10 * p["v"] * np.log10(d / p["d0"])
    W = p.get("W", 0)
    if W:
        # регулярна сітка кімнат зі стороною W: кількість стін між користувачем і станцією
        cu = np.floor(users / W)
        cb = np.floor(np.minimum(bs, p["A"] - 1e-9) / W)
        if bs.ndim == 3:
            nw = np.abs(cu[None, :, None, :] - cb[:, None, :, :]).sum(-1)
        else:
            nw = np.abs(cu[:, None, :] - cb[None, :, :]).sum(-1)
        pl_db = pl_db + nw * p["Lw"]
    return 10 ** ((p["P_bs_dbm"] + p["G_bs_dbi"] - pl_db - 30) / 10)

def rates(users, bs, p, interference=True, share=False):
    """Швидкості користувачів за (3). interference=False: знаменник містить лише N0 (редукований опис, delta=0).
    share=True: ресурс БС ділиться між прикріпленими користувачами (варіант для аналізу чутливості)."""
    P = rx_power_w(users, bs, p)                 # (...,N,M)
    j = P.argmax(-1)                             # обслуговуюча БС
    S = np.take_along_axis(P, j[..., None], -1)[..., 0]
    I = P.sum(-1) - S if interference else 0.0
    sinr = S / (I + noise_w(p))
    R = p["B"] * np.log2(1 + sinr)
    M = P.shape[-1]
    load = np.stack([(j == m).sum(-1) for m in range(M)], -1)   # (...,M)
    if share:
        R = R / np.take_along_axis(load, j, -1)
    return R, load

def fitness(users, bs, p, interference=True, share=False):
    """Цільова функція (2): середня швидкість мінус штрафи. Векторизовано за першою віссю bs (популяція)."""
    R, load = rates(users, bs, p, interference, share)
    N = users.shape[0]
    if p.get("obj") == "cov":
        # критерій покриття: мінімізувати частку користувачів з R < R_min, середня швидкість лише розв'язує нічиї
        f = -(R < p["R_min"]).mean(-1) + 1e-10 * R.mean(-1)
        b = bs if bs.ndim == 3 else bs[None]
        dd = np.sqrt(((b[:, :, None, :] - b[:, None, :, :]) ** 2).sum(-1)); M = b.shape[1]
        iu = np.triu_indices(M, 1); close = (dd[:, iu[0], iu[1]] < p["d_min"]).sum(-1)
        return f - 0.01 * (close if bs.ndim == 3 else close[0])
    f = R.mean(-1)
    f = f - p["alpha"] * (R < p["R_min"]).sum(-1) / N
    f = f - p["beta"] * (load > p["n_max"]).sum(-1)
    b = bs if bs.ndim == 3 else bs[None]
    dd = np.sqrt(((b[:, :, None, :] - b[:, None, :, :]) ** 2).sum(-1))
    M = b.shape[1]
    iu = np.triu_indices(M, 1)
    close = (dd[:, iu[0], iu[1]] < p["d_min"]).sum(-1)
    f = f - p["gamma"] * (close if bs.ndim == 3 else close[0])
    return f

def ga(users, M, p, interference=True, share=False, pop=40, gens=150, k_tour=3, elite=2,
       sigma=5.0, pm=0.3, rng=None, grid=None):
    """ГА за (6)-(8): турнірний відбір з елітизмом, змішувальне схрещування, гаусова мутація однієї БС.
    grid: якщо задано крок сітки, координати округлюються до вузлів (дискретний варіант)."""
    rng = rng or np.random.default_rng()
    A = p["A"]
    def snap(X):
        if grid is None:
            return np.clip(X, 0, A)
        g0 = grid / 2
        return np.clip(np.round((X - g0) / grid) * grid + g0, g0, A - g0)
    X = snap(rng.uniform(0, A, (pop, M, 2)))
    fit = fitness(users, X, p, interference, share)
    best_hist = []
    for _ in range(gens):
        order = np.argsort(-fit)
        new = [X[order[i]] for i in range(elite)]
        while len(new) < pop:
            a = rng.choice(pop, k_tour); b = rng.choice(pop, k_tour)
            p1 = X[a[np.argmax(fit[a])]]; p2 = X[b[np.argmax(fit[b])]]
            lam = rng.uniform()
            c = lam * p1 + (1 - lam) * p2                      # (7)
            if rng.uniform() < pm:
                jm = rng.integers(M); c = c.copy(); c[jm] += rng.normal(0, sigma, 2)   # (8)
            new.append(snap(c))
        X = np.stack(new)
        fit = fitness(users, X, p, interference, share)
        best_hist.append(fit.max())
    i = fit.argmax()
    return X[i], fit[i], np.array(best_hist)
