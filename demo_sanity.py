"""
Sanity demo (NOT paper results): synthetic non-IID IIoT-like data, logistic-regression IDS,
10 sites, plain FedAvg vs Faithful Aggregation, with and without Byzantine sites.
Fixed seeds; deterministic.
"""
import time
import numpy as np
import faithful_agg as fa

N_SITES, N_FEAT, ROUNDS, LR, LOCAL_STEPS = 10, 20, 15, 0.5, 5
SEED = 11


def make_data(rng):
    w_true = rng.normal(size=N_FEAT)
    sites = []
    for k in range(N_SITES):
        shift = rng.normal(scale=0.5, size=N_FEAT)          # per-site feature shift (non-IID)
        p_attack = rng.uniform(0.15, 0.6)                    # per-site class skew
        n = 300
        y = (rng.random(n) < p_attack).astype(float)
        X = rng.normal(size=(n, N_FEAT)) + shift + np.outer(y, w_true) * 0.8
        sites.append((np.c_[X, np.ones(n)], y))
    y = (rng.random(2000) < 0.35).astype(float)
    Xt = rng.normal(size=(2000, N_FEAT)) + np.outer(y, w_true) * 0.8
    return sites, (np.c_[Xt, np.ones(2000)], y)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def local_update(w, X, y):
    w0 = w.copy()
    for _ in range(LOCAL_STEPS):
        w = w - LR * X.T @ (sigmoid(X @ w) - y) / len(y)
    return w - w0


def acc(w, test):
    X, y = test
    return float(np.mean((sigmoid(X @ w) > 0.5) == y))


def attack(kind, u, rng):
    if kind == "signflip":
        return -4.0 * u
    if kind == "scale":
        return 25.0 * u
    if kind == "noise":
        return rng.normal(size=u.shape) * 2.0
    return u


def run(mode, n_byz, kind, B, tau, seed=SEED):
    rng = np.random.default_rng(seed)
    sites, test = make_data(rng)
    crng = np.random.default_rng(seed + 1)
    w = np.zeros(N_FEAT + 1)
    ref = None
    byz = set(range(n_byz))
    led = fa.Ledger()
    rej_h = rej_b = 0
    for r in range(ROUNDS):
        ups = []
        for k, (X, y) in enumerate(sites):
            u = local_update(w, X, y)
            ups.append(attack(kind, u, crng) if k in byz else u)
        if mode == "fedavg":
            agg = np.mean(ups, axis=0)
        else:
            subs = [fa.make_submission(k, u, crng, with_commitments=True) for k, u in enumerate(ups)]
            res = fa.run_round(subs, crng, B, tau, ref, ledger=led)
            assert fa.audit(res.transcript)
            agg, ref = res.mean_update, (res.mean_update if res.n_accepted else ref)
            rej_b += sum(1 for k in byz if not res.bits[k])
            rej_h += sum(1 for k in range(N_SITES) if k not in byz and not res.bits[k])
        w = w + agg
    return acc(w, test), rej_h, rej_b, led


if __name__ == "__main__":
    # calibrate public bound B from a clean FedAvg run's honest update norms (demo only)
    rng = np.random.default_rng(SEED)
    sites, _ = make_data(rng)
    w = np.zeros(N_FEAT + 1)
    norms = []
    for _ in range(ROUNDS):
        ups = [local_update(w, X, y) for X, y in sites]
        norms += [np.linalg.norm(u) for u in ups]
        w = w + np.mean(ups, axis=0)
    B = 1.5 * max(norms)
    TAU = 0.0   # norm bound + non-negative alignment with the previous aggregate
    print(f"public bound B = {B:.3f}, tau = {TAU}\n")
    print(f"{'scenario':34s} {'FedAvg':>8s} {'Faithful':>9s} {'honest rej':>11s} {'byz rej':>8s}")
    for label, nb, kind in [("no attack", 0, "none"),
                            ("2/10 sign-flip x4", 2, "signflip"),
                            ("2/10 scaled x25", 2, "scale"),
                            ("3/10 random noise", 3, "noise")]:
        a_plain = run("fedavg", nb, kind, B, TAU)[0]
        a_fa, rh, rb, led = run("faithful", nb, kind, B, TAU)
        tot_h = (N_SITES - nb) * ROUNDS
        print(f"{label:34s} {a_plain:8.3f} {a_fa:9.3f} {rh:5d}/{tot_h:<5d} {rb:3d}/{nb * ROUNDS:<4d}")

    print("\nCost per round (d = 21, 10 sites; communication is analytic, time is simulation CPU):")
    a, rh, rb, led = run("faithful", 0, "none", B, TAU)
    for ph in ["upload", "secure_tests", "aggregate", "publish"]:
        print(f"  {ph:13s} {led.bytes_[ph] / ROUNDS / 1024:9.1f} KiB  rounds/round {led.rounds[ph] / ROUNDS:5.1f}")
    print(f"  offline (triples) {led.bytes_['offline'] / ROUNDS / 1024:9.1f} KiB")
    plain = N_SITES * (N_FEAT + 1) * 8
    print(f"  plain FedAvg upload would be {plain / 1024:.1f} KiB/round")

    print("\nScaling with model dimension d (one round, 10 sites, no cosine ref):")
    print(f"  {'d':>6s} {'KiB/round':>10s} {'sim secs':>9s}")
    for d in (21, 100, 400):
        rng = np.random.default_rng(3)
        ups = [rng.normal(size=d) * 0.05 for _ in range(10)]
        led = fa.Ledger()
        t0 = time.perf_counter()
        subs = [fa.make_submission(k, u, rng) for k, u in enumerate(ups)]
        fa.run_round(subs, rng, B=5.0, ledger=led)
        print(f"  {d:6d} {led.total_bytes() / 1024:10.1f} {time.perf_counter() - t0:9.2f}")
