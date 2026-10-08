import numpy as np
import faithful_agg as fa

rng = np.random.default_rng(7)
D = 24


def rnd(norm):
    v = rng.normal(size=D)
    return v / np.linalg.norm(v) * norm


def test_encode_roundtrip():
    x = rng.normal(size=D) * 3
    assert np.allclose(fa.dec(fa.enc(x)), x, atol=2 ** -fa.FBITS)


def test_sharing_reconstructs():
    x = fa.enc(rng.normal(size=D))
    s1, s2 = fa.share_vec(x, rng)
    assert fa.vadd(s1, s2) == x
    assert s1 != x and s2 != x


def test_secure_norm_matches_plaintext():
    dealer = fa.Dealer(rng)
    for _ in range(20):
        u = rng.normal(size=D) * rng.uniform(0.1, 20)
        s1, s2 = fa.share_vec(fa.enc(u), rng)
        t1, t2 = dealer.norm_triple(D)
        z1, z2 = fa.secure_norm_sq(s1, s2, t1, t2)
        got = fa.signed(z1 + z2) / fa.SCALE ** 2
        want = float(np.sum((np.rint(u * fa.SCALE) / fa.SCALE) ** 2))
        assert abs(got - want) < 1e-9 * max(1, want)


def test_secure_square_matches_plaintext():
    dealer = fa.Dealer(rng)
    t = int(rng.integers(-10 ** 6, 10 ** 6))
    a1 = fa.rand_fe(rng)
    t1, t2 = a1, (t - a1) % fa.Q
    s1, s2 = dealer.scalar_triple()
    z1, z2 = fa.secure_square(t1, t2, s1, s2)
    assert fa.signed(z1 + z2) == t * t


def test_accept_bits_match_plaintext_rule():
    ref = rng.normal(size=D)
    B, tau = 2.0, 0.3
    subs, us = [], []
    for k in range(40):
        u = rnd(rng.uniform(0.2, 4.0))
        if rng.random() < 0.5:                       # bias toward reference
            u = 0.6 * u + 0.8 * ref / np.linalg.norm(ref) * np.linalg.norm(u)
        us.append(u)
        subs.append(fa.make_submission(k, u, rng, with_commitments=False))
    res = fa.run_round(subs, rng, B, tau, ref, verify_openings=False)
    checked = 0
    for u, b in zip(us, res.bits):
        dp = float(u @ ref) / (np.linalg.norm(ref) * np.linalg.norm(u))
        margin = min(abs(np.linalg.norm(u) - B), abs(dp - tau), abs(dp))
        if margin > 0.02:                             # skip razor-edge cases
            assert b == fa.plaintext_rule(u, ref, B, tau)
            checked += 1
    assert checked >= 25


def test_negative_tau_matches_plaintext_rule():
    ref = rng.normal(size=D)
    B, tau = 2.0, -0.3
    subs, us = [], []
    for k in range(60):
        u = rnd(rng.uniform(0.2, 3.0))
        u = u + rng.uniform(-1.2, 1.2) * ref / np.linalg.norm(ref) * np.linalg.norm(u)
        us.append(u)
        subs.append(fa.make_submission(k, u, rng, with_commitments=False))
    res = fa.run_round(subs, rng, B, tau, ref, verify_openings=False)
    checked = 0
    for u, b in zip(us, res.bits):
        dp = float(u @ ref) / (np.linalg.norm(ref) * np.linalg.norm(u))
        if min(abs(np.linalg.norm(u) - B), abs(dp - tau), abs(dp)) > 0.02:
            assert b == fa.plaintext_rule(u, ref, B, tau)
            checked += 1
    assert checked >= 35 and 0 < sum(res.bits) < len(res.bits)


def test_aggregate_equals_plaintext_mean_of_accepted():
    ups = [rnd(0.5) for _ in range(6)] + [rnd(50.0)]   # last one violates norm bound
    subs = [fa.make_submission(k, u, rng) for k, u in enumerate(ups)]
    res = fa.run_round(subs, rng, B=2.0)
    assert res.bits == [1] * 6 + [0]
    assert np.allclose(res.mean_update, np.mean(ups[:6], axis=0), atol=1e-4)
    assert fa.audit(res.transcript)


def test_merkle_inclusion_and_tamper():
    subs = [fa.make_submission(k, rnd(0.5), rng) for k in range(5)]
    res = fa.run_round(subs, rng, B=2.0)
    for j in range(2):
        for i in range(5):
            ok, bit = fa.site_verify(res.transcript, i, j)
            assert ok and bit == 1
    tr = res.transcript
    path = fa.merkle_path(list(tr.commits[0]), 2)
    assert not fa.merkle_verify(tr.commits[0][2] + 1, path, tr.roots[0])


def test_server_dropping_accepted_site_is_detected():
    subs = [fa.make_submission(k, rnd(0.5), rng) for k in range(5)]
    res = fa.run_round(subs, rng, B=2.0)
    tr = res.transcript
    # S2 silently leaves site 3's share out of its aggregate
    bad_agg = fa.vsub(tr.agg_share[1], subs[3].shares[1])
    bad_rand = (tr.agg_rand[1] - subs[3].rand[1]) % fa.Q
    forged = fa.Transcript(tr.roots, tr.commits, tr.bitmap,
                           (tr.agg_share[0], bad_agg), (tr.agg_rand[0], bad_rand), tr.site_ids)
    assert not fa.audit(forged)
    # S2 flips the bit instead: bitmaps disagree
    bm = [list(tr.bitmap[0]), list(tr.bitmap[1])]
    bm[1][3] = 0
    forged2 = fa.Transcript(tr.roots, tr.commits, (bm[0], bm[1]), tr.agg_share, tr.agg_rand, tr.site_ids)
    assert not fa.audit(forged2)


def test_inconsistent_share_commitment_rejected():
    subs = [fa.make_submission(k, rnd(0.5), rng) for k in range(4)]
    s = subs[1]
    bad = list(s.shares[0])
    bad[0] = (bad[0] + 1) % fa.Q                      # share no longer matches commitment
    subs[1] = fa.Submission(s.site_id, (bad, s.shares[1]), s.rand, s.commit)
    res = fa.run_round(subs, rng, B=2.0)
    assert res.bits == [1, 0, 1, 1]


def test_range_proof_blocks_overflow_attack():
    huge = np.zeros(D)
    huge[0] = 1e30
    subs = [fa.make_submission(0, huge, rng), fa.make_submission(1, rnd(0.5), rng)]
    res = fa.run_round(subs, rng, B=2.0)
    assert res.bits == [0, 1]


def test_no_accepted_sites_gives_zero_update():
    subs = [fa.make_submission(k, rnd(40.0), rng, False) for k in range(3)]
    res = fa.run_round(subs, rng, B=1.0, verify_openings=False)
    assert res.n_accepted == 0 and np.allclose(res.mean_update, 0)


if __name__ == "__main__":
    import sys
    import traceback
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except Exception:
                fails += 1
                print("FAIL", name)
                traceback.print_exc()
    sys.exit(1 if fails else 0)
