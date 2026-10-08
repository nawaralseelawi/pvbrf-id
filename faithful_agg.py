"""
faithful_agg.py -- simulation core for cryptographic Faithful Aggregation.

Two non-colluding servers (S1, S2) jointly aggregate model updates from N sites.

  Blindness           additive secret sharing over GF(Q), Q = 2^127 - 1
  Robustness          secure squared-norm and cosine tests (Beaver triples);
                      only ONE accept bit per site per round is revealed
  Proof of Inclusion  per-site vector-Pedersen commitments to each share,
                      Merkle root per server, accept bitmap, and a
                      homomorphic check that each server's published aggregate
                      share equals the sum of the committed shares it accepted

WHAT IS REAL vs. IDEAL (state this in the paper):
  real   : secret sharing, fixed-point encoding, Beaver inner products,
           Merkle tree (SHA-256), Pedersen commitments + homomorphic audit
  ideal  : (1) secure comparison -> reveals only the accept bit; its cost is a
               parameter (CMP_ELEMS / CMP_ROUNDS), not measured
           (2) per-coordinate range proofs on site inputs (|x| <= XMAX)
           (3) Beaver-triple generation (trusted dealer; offline cost counted)
  toy    : Pedersen group is 512-bit (cost scaling only, NOT secure); use a
           256-bit elliptic curve for real deployments. Randomness is a seeded
           PRNG for reproducibility; use a CSPRNG in practice.
"""
import hashlib
import time
from dataclasses import dataclass, field
import numpy as np

# ----------------------------------------------------------------- parameters
Q = (1 << 127) - 1            # field modulus == Pedersen subgroup order
HALF = Q // 2
FBITS = 16                    # fixed-point fractional bits
SCALE = 1 << FBITS
FE_BYTES = 16                 # bytes per field element (127 bits)
XMAX = float(1 << 20)         # per-coordinate bound enforced by (ideal) range proof
# Toy 512-bit prime P = k*Q + 1 (subgroup of order Q)
P = 13407807929942597099574024998205846127400561808199604419299003363521476802463387498260943404831006840881540414643545053962531937640940488478187180634668743
GROUP_ELEM_BYTES = 32         # reported cost assumes a 256-bit EC point, not the toy group
# Analytic cost of one ideal secure comparison (per direction): placeholder estimate
CMP_ELEMS = 2 * 127
CMP_ROUNDS = 8


# ------------------------------------------------------------ field utilities
def enc(x, power=1):
    """float array -> list of field elements at scale 2^(FBITS*power)."""
    x = np.asarray(x, dtype=np.float64)
    ints = np.rint(x * float(SCALE ** power)).astype(np.int64) if power == 1 else \
        [int(round(float(v) * SCALE ** power)) for v in x]
    return [int(v) % Q for v in ints]


def signed(v):
    v %= Q
    return v - Q if v > HALF else v


def dec(vec, power=1):
    return np.array([signed(v) / float(SCALE ** power) for v in vec])


def vadd(a, b):
    return [(x + y) % Q for x, y in zip(a, b)]


def vsub(a, b):
    return [(x - y) % Q for x, y in zip(a, b)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b)) % Q


def rand_vec(rng, d):
    raw = rng.bytes(FE_BYTES * d)
    return [int.from_bytes(raw[FE_BYTES * i:FE_BYTES * (i + 1)], "big") % Q for i in range(d)]


def rand_fe(rng):
    return int.from_bytes(rng.bytes(FE_BYTES), "big") % Q


def share_vec(x, rng):
    s1 = rand_vec(rng, len(x))
    return s1, vsub(x, s1)


# ------------------------------------------------------------------- ledger
@dataclass
class Ledger:
    bytes_: dict = field(default_factory=dict)
    rounds: dict = field(default_factory=dict)
    secs: dict = field(default_factory=dict)

    def add(self, phase, nbytes=0, rounds=0, secs=0.0):
        self.bytes_[phase] = self.bytes_.get(phase, 0) + nbytes
        self.rounds[phase] = self.rounds.get(phase, 0) + rounds
        self.secs[phase] = self.secs.get(phase, 0.0) + secs

    def total_bytes(self, exclude=("offline",)):
        return sum(v for k, v in self.bytes_.items() if k not in exclude)


# --------------------------------------------------------------- Pedersen
_GENS = {}


def _gen(label, idx):
    key = (label, idx)
    if key not in _GENS:
        h = int.from_bytes(hashlib.sha512(label + idx.to_bytes(4, "big")).digest(), "big") % P
        g = pow(h, (P - 1) // Q, P)
        assert g != 1
        _GENS[key] = g
    return _GENS[key]


def pedersen_commit(vec, r):
    acc = pow(_gen(b"H", 0), r % Q, P)
    for k, x in enumerate(vec):
        if x:
            acc = acc * pow(_gen(b"G", k), x, P) % P
    return acc


def pedersen_product(commits):
    acc = 1
    for c in commits:
        acc = acc * c % P
    return acc


# ------------------------------------------------------------------ Merkle
def _h(b):
    return hashlib.sha256(b).digest()


def _leaf(c):
    return _h(b"\x00" + c.to_bytes(64, "big"))


def merkle_levels(commits):
    level = [_leaf(c) for c in commits]
    levels = [level]
    while len(level) > 1:
        if len(level) % 2:
            level = level + [level[-1]]
        level = [_h(b"\x01" + level[i] + level[i + 1]) for i in range(0, len(level), 2)]
        levels.append(level)
    return levels


def merkle_root(commits):
    return merkle_levels(commits)[-1][0]


def merkle_path(commits, idx):
    path = []
    levels = merkle_levels(commits)
    for level in levels[:-1]:
        lv = level + [level[-1]] if len(level) % 2 else level
        sib = idx ^ 1
        path.append((lv[sib], sib > idx))
        idx //= 2
    return path


def merkle_verify(commit, path, root):
    node = _leaf(commit)
    for sib, sib_is_right in path:
        node = _h(b"\x01" + (node + sib if sib_is_right else sib + node))
    return node == root


# -------------------------------------------------- submissions & dealer
@dataclass
class Submission:
    site_id: int
    shares: tuple          # (s1, s2) field vectors
    rand: tuple            # Pedersen randomness per server
    commit: tuple          # public commitments (C1, C2)
    range_ok: bool = True


def make_submission(site_id, update, rng, with_commitments=True):
    update = np.asarray(update, dtype=np.float64)
    range_ok = bool(np.all(np.abs(update) <= XMAX))      # ideal range proof
    x = enc(np.clip(update, -XMAX, XMAX))
    s1, s2 = share_vec(x, rng)
    r = (rand_fe(rng), rand_fe(rng))
    c = (pedersen_commit(s1, r[0]), pedersen_commit(s2, r[1])) if with_commitments else (0, 0)
    return Submission(site_id, (s1, s2), r, c, range_ok)


class Dealer:
    """Trusted dealer for the offline phase (ideal)."""

    def __init__(self, rng):
        self.rng = rng

    def norm_triple(self, d):
        a = rand_vec(self.rng, d)
        a1, a2 = share_vec(a, self.rng)
        c = dot(a, a)
        c1 = rand_fe(self.rng)
        return (a1, c1), (a2, (c - c1) % Q)

    def scalar_triple(self):
        a = rand_fe(self.rng)
        a1 = rand_fe(self.rng)
        c = a * a % Q
        c1 = rand_fe(self.rng)
        return (a1, c1), ((a - a1) % Q, (c - c1) % Q)


# ------------------------------------------------- secure sub-protocols
def secure_norm_sq(x1, x2, tri1, tri2):
    """Shares of ||x||^2 (scale 2^(2F)); opens one masked vector e = x - a."""
    (a1, c1), (a2, c2) = tri1, tri2
    e = vadd(vsub(x1, a1), vsub(x2, a2))                 # opened to both servers
    z1 = (c1 + 2 * dot(e, a1) + dot(e, e)) % Q
    z2 = (c2 + 2 * dot(e, a2)) % Q
    return z1, z2


def secure_square(t1, t2, tri1, tri2):
    (a1, c1), (a2, c2) = tri1, tri2
    e = (t1 + t2 - a1 - a2) % Q
    return (c1 + 2 * e * a1 + e * e) % Q, (c2 + 2 * e * a2) % Q


def ideal_accept(z_shares_list):
    """Ideal functionality: reveals ONLY the AND of (z >= 0) over the inputs."""
    for z1, z2 in z_shares_list:
        if signed(z1 + z2) < 0:
            return 0
    return 1


def ideal_accept_tolerant(z_norm, z_dp, z_tol):
    """Ideal functionality for tau < 0 (cos >= tau): norm_ok AND (dp >= 0 OR dp^2 <= tau^2 ||x||^2).
    Reveals ONLY the final accept bit."""
    ok_norm = signed(z_norm[0] + z_norm[1]) >= 0
    ok_dp = signed(z_dp[0] + z_dp[1]) >= 0
    ok_tol = signed(z_tol[0] + z_tol[1]) >= 0
    return int(ok_norm and (ok_dp or ok_tol))


# -------------------------------------------------------------- protocol
@dataclass
class Transcript:
    roots: tuple
    commits: tuple          # per server: list of commitments, one per site
    bitmap: tuple           # per server: accepted bits (must match)
    agg_share: tuple        # per server: published aggregate share (vector)
    agg_rand: tuple         # per server: summed randomness
    site_ids: list


@dataclass
class RoundResult:
    mean_update: np.ndarray
    bits: list
    n_accepted: int
    transcript: Transcript
    ledger: Ledger


def run_round(submissions, rng, B, tau=None, ref=None, verify_openings=True,
              dealer=None, ledger=None):
    """
    One aggregation round.
      B    : public L2 bound on an accepted update
      tau  : public cosine threshold against `ref` (public previous aggregate)
      ref  : public reference direction (float vector) or None (skip cosine test)
    """
    led = ledger or Ledger()
    dealer = dealer or Dealer(rng)
    n, d = len(submissions), len(submissions[0].shares[0])

    # -- phase 1: upload (site -> servers, commitments -> bulletin board)
    led.add("upload", n * (2 * d * FE_BYTES + 2 * FE_BYTES + 2 * GROUP_ELEM_BYTES), rounds=1)

    # -- phase 2: servers privately check share/commitment openings (real)
    t0 = time.perf_counter()
    opening_ok = []
    for sub in submissions:
        ok = sub.range_ok
        if verify_openings and ok:
            for j in range(2):
                ok &= pedersen_commit(sub.shares[j], sub.rand[j]) == sub.commit[j]
        opening_ok.append(bool(ok))
    led.add("open_check", 0, 0, time.perf_counter() - t0)

    # -- phase 3: secure tests on shares
    use_cos = ref is not None and tau is not None and np.linalg.norm(ref) > 0
    if use_cos:
        r_hat = enc(np.asarray(ref) / np.linalg.norm(ref))
    B2 = int(round(B * B * SCALE ** 2)) % Q
    tau2c = int(round((tau ** 2) * SCALE ** 2)) % Q if use_cos else 0

    t0 = time.perf_counter()
    bits = []
    for sub, ok in zip(submissions, opening_ok):
        if not ok:
            bits.append(0)
            continue
        x1, x2 = sub.shares
        t1, t2 = dealer.norm_triple(d)
        n1, n2 = secure_norm_sq(x1, x2, t1, t2)
        zs = [((B2 - n1) % Q, (-n2) % Q)]                      # B^2 - ||x||^2 >= 0
        if use_cos:
            d1, d2 = dot(x1, r_hat), dot(x2, r_hat)            # local on shares
            s1_, s2_ = dealer.scalar_triple()
            q1, q2 = secure_square(d1, d2, s1_, s2_)
            if tau >= 0:
                zs.append(((q1 - tau2c * n1) % Q, (q2 - tau2c * n2) % Q))   # <x,r>^2 >= tau^2||x||^2
                zs.append((d1, d2))                                          # <x,r> >= 0
            else:   # tolerant cone: accept also slightly negative alignment, cos >= tau
                bits.append(ideal_accept_tolerant(zs[0], (d1, d2),
                                                  ((tau2c * n1 - q1) % Q, (tau2c * n2 - q2) % Q)))
                continue
        bits.append(ideal_accept(zs))
    led.add("secure_tests", 0, 0, time.perf_counter() - t0)
    n_ok = sum(opening_ok)
    # analytic communication: norm opening (d elems each way), square opening, comparisons
    led.add("secure_tests", n_ok * 2 * d * FE_BYTES, rounds=1)
    if use_cos:
        led.add("secure_tests", n_ok * 2 * 2 * FE_BYTES, rounds=1)
    led.add("secure_tests", n_ok * (3 if use_cos else 1) * 2 * CMP_ELEMS * FE_BYTES, rounds=CMP_ROUNDS)
    led.add("offline", n_ok * (2 * (d + 1) + (2 * 2 if use_cos else 0)) * FE_BYTES)

    # -- phase 4: aggregate accepted shares, publish transcript
    A, R = [], []
    for j in range(2):
        acc = [0] * d
        rr = 0
        for sub, b in zip(submissions, bits):
            if b:
                acc = vadd(acc, sub.shares[j])
                rr = (rr + sub.rand[j]) % Q
        A.append(acc)
        R.append(rr)
    count = sum(bits)
    total = vadd(A[0], A[1])
    mean = dec(total) / count if count else np.zeros(d)
    led.add("aggregate", 2 * (d * FE_BYTES + FE_BYTES), rounds=1)

    commits = tuple([s.commit[j] for s in submissions] for j in range(2))
    roots = tuple(merkle_root(c) for c in commits)
    tr = Transcript(roots, commits, (list(bits), list(bits)), tuple(A), tuple(R),
                    [s.site_id for s in submissions])
    led.add("publish", 2 * (32 + n // 8 + 1), rounds=1)
    return RoundResult(mean, bits, count, tr, led)


# ------------------------------------------------------------ verification
def site_verify(tr, idx, server):
    """Site `idx` checks its commitment is in server's published Merkle root
    and reads its accept bit."""
    path = merkle_path(list(tr.commits[server]), idx)
    return merkle_verify(tr.commits[server][idx], path, tr.roots[server]), tr.bitmap[server][idx]


def audit(tr):
    """Public audit: bitmaps agree and each server's published aggregate share
    equals (homomorphically) the sum of the committed shares it accepted."""
    if tr.bitmap[0] != tr.bitmap[1]:
        return False
    for j in range(2):
        acc = [c for c, b in zip(tr.commits[j], tr.bitmap[j]) if b]
        lhs = pedersen_product(acc)
        rhs = pedersen_commit(tr.agg_share[j], tr.agg_rand[j])
        if lhs != rhs:
            return False
    return True


# ------------------------------------------------ plaintext reference rule
def plaintext_rule(u, ref, B, tau):
    u = np.asarray(u, dtype=np.float64)
    if np.linalg.norm(u) > B:
        return 0
    if ref is not None and tau is not None and np.linalg.norm(ref) > 0:
        dp = float(u @ (ref / np.linalg.norm(ref)))
        if tau >= 0:
            if dp < 0 or dp * dp < tau * tau * float(u @ u):
                return 0
        elif not (dp >= 0 or dp * dp <= tau * tau * float(u @ u)):
            return 0
    return 1
