"""
Sanity experiment (NOT final paper results): reference source x cosine threshold x attack.
  ref = 'prev' : previous public aggregate (cold start: no reference in round 1)
  ref = 'root' : update direction from a small clean server-side root set (FLTrust-style),
                 known to the servers, available from round 1
Attacks (2 of 10 sites):
  signflip : -4 * honest update
  adaptive : worst case -- attacker knows B, tau and the reference; submits a vector of norm
             0.95*B at the minimum allowed cosine to the reference, with the remainder along a
             fixed direction orthogonal to it (steady drift)
"""
import numpy as np
import faithful_agg as fa
from demo_sanity import (make_data, local_update, acc, N_SITES, N_FEAT, ROUNDS, SEED)

N_BYZ = 2


def run(ref_mode, tau, attack, B, seed=SEED):
    rng = np.random.default_rng(seed)
    sites, test = make_data(rng)
    Xt, yt = test
    root, evalset = (Xt[:300], yt[:300]), (Xt[300:], yt[300:])
    crng = np.random.default_rng(seed + 1)
    drift = crng.normal(size=N_FEAT + 1)
    w, prev = np.zeros(N_FEAT + 1), None
    rej_h = rej_b = 0
    for r in range(ROUNDS):
        ref = prev if ref_mode == "prev" else local_update(w, *root)
        ups = []
        for k, (X, y) in enumerate(sites):
            u = local_update(w, X, y)
            if k < N_BYZ:
                if attack == "signflip":
                    u = -4.0 * u
                elif attack == "adaptive":
                    if ref is not None and np.linalg.norm(ref) > 0:
                        rh = ref / np.linalg.norm(ref)
                        p = drift - (drift @ rh) * rh
                        p /= np.linalg.norm(p)
                        c = min(max(tau, 0.0) + 0.02, 0.999)
                        u = 0.95 * B * (c * rh + np.sqrt(1 - c * c) * p)
                    else:
                        u = 0.95 * B * drift / np.linalg.norm(drift)
            ups.append(u)
        subs = [fa.make_submission(k, u, crng, with_commitments=False) for k, u in enumerate(ups)]
        res = fa.run_round(subs, crng, B, tau, ref, verify_openings=False)
        rej_b += sum(1 for k in range(N_BYZ) if not res.bits[k])
        rej_h += sum(1 for k in range(N_BYZ, N_SITES) if not res.bits[k])
        if res.n_accepted:
            w = w + res.mean_update
            prev = res.mean_update
    return acc(w, evalset), rej_h, rej_b


def fedavg(attack, B, seed=SEED):
    # undefended reference under the same attacks (adaptive here = norm 0.95B along a fixed direction)
    rng = np.random.default_rng(seed)
    sites, test = make_data(rng)
    Xt, yt = test
    evalset = (Xt[300:], yt[300:])
    crng = np.random.default_rng(seed + 1)
    drift = crng.normal(size=N_FEAT + 1)
    w = np.zeros(N_FEAT + 1)
    for _ in range(ROUNDS):
        ups = []
        for k, (X, y) in enumerate(sites):
            u = local_update(w, X, y)
            if k < N_BYZ:
                u = -4.0 * u if attack == "signflip" else 0.95 * B * drift / np.linalg.norm(drift)
            ups.append(u)
        w = w + np.mean(ups, axis=0)
    return acc(w, evalset)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    sites, _ = make_data(rng)
    w, norms = np.zeros(N_FEAT + 1), []
    for _ in range(ROUNDS):
        ups = [local_update(w, X, y) for X, y in sites]
        norms += [np.linalg.norm(u) for u in ups]
        w = w + np.mean(ups, axis=0)
    B = 1.5 * max(norms)
    honest_total = (N_SITES - N_BYZ) * ROUNDS
    byz_total = N_BYZ * ROUNDS
    print(f"B = {B:.3f}; honest updates per run = {honest_total}, attacker updates = {byz_total}\n")
    for attack in ("signflip", "adaptive"):
        print(f"== attack: {attack}   (undefended FedAvg accuracy: {fedavg(attack, B):.3f})")
        print(f"{'ref':6s} {'tau':>5s} {'acc':>7s} {'honest rejected':>16s} {'attackers rejected':>19s}")
        for ref_mode in ("prev", "root"):
            for tau in (0.0, 0.2, 0.4):
                a, rh, rb = run(ref_mode, tau, attack, B)
                print(f"{ref_mode:6s} {tau:5.1f} {a:7.3f} {rh:7d}/{honest_total:<8d} {rb:9d}/{byz_total:<8d}")
        print()
