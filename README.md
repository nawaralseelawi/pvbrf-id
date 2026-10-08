# PVBRF-ID: Private, Verifiable and Byzantine-Robust Federated Intrusion Detection for Oilfield Enterprise Networks

Simulation code, raw results and figure scripts for the paper

> Mustafa S. Aljumaily, Ali Ataeemh Allami, Mustafa N. Mnati, Nawar S. Alseelawi, Shahab Abdulla,
> "PVBRF-ID: Private, Verifiable and Byzantine-Robust Federated Intrusion Detection for Oilfield
> Enterprise Networks," arXiv preprint, 2026. (arXiv link to be added after announcement.)

PVBRF-ID is a two-server aggregator for federated learning. Sites secret-share their model updates;
the servers jointly test each update's norm and its cosine similarity to a clean root-set reference
inside the shares, reveal one accept bit per site, and publish commitments, Merkle roots and a
homomorphic audit so that inclusion of accepted updates is verifiable.

> **Scope.** This is a simulation. Secure comparison, per-coordinate range proofs, Beaver-triple
> generation and secure clipping are modelled as ideal functionalities; the Pedersen group is a toy
> 512-bit group used only for cost scaling. Experiments use the public Edge-IIoTset traffic with
> simulated sites. No operator data were used. Read the paper's limitations before quoting a number.

## Layout

| path | purpose |
|---|---|
| `faithful_agg.py` | protocol core: additive secret sharing over GF(2^127-1), fixed-point encoding, Beaver-triple norm and cosine tests (including the tolerant case tau < 0), SHA-256 Merkle inclusion, Pedersen commitments and the homomorphic audit |
| `test_core.py` | 12 correctness and tamper-detection tests (`python test_core.py`) |
| `demo_sanity.py` | end-to-end sanity run on synthetic data plus a cost benchmark |
| `exp_ref_tau.py` | reference-source x cosine-threshold x attack sweep on synthetic data (single seed, sanity only) |
| `notebooks/faithful_agg_edgeiiot_v5_final.ipynb` | the main experiment: 870 runs (10 seeds x 87 configurations) on Edge-IIoTset |
| `notebooks/faithful_agg_edgeiiot_v5b_tolerance_attack.ipynb` | the supplementary experiment: 160 runs with the tolerance-exploiting adaptive attacker |
| `results/v5_full/` | raw per-run output of the main experiment (`results.csv`) and all derived tables |
| `results/v5b_tolerance/` | raw per-run output of the supplementary experiment |
| `make_figures.py` | regenerates manuscript Figures 2-5 and Supplementary Table S3 from the two `results.csv` files |
| `figures/` | the manuscript figures |

The code and result files keep the development name of the method, `faithful_agg` / `faithful_root`;
`faithful_root_t-0.3` in `results.csv` is PVBRF-ID with the root-set reference and tau = -0.3.

## Reproduce

```bash
pip install -r requirements.txt
python test_core.py                 # 12 tests, about a minute
python make_figures.py              # figures from the shipped results, seconds
```

To re-run the experiments, download the Edge-IIoTset DNN file (`DNN-EdgeIIoT-dataset.csv`,
2,219,201 flows; available from the dataset authors via IEEE DataPort or Kaggle), place it at
`data/DNN-EdgeIIoT-dataset.csv`, and run the two notebooks from the `notebooks/` folder in order:

1. `faithful_agg_edgeiiot_v5_final.ipynb` (about 1.1-2.9 h on 8 cores; resumable; writes `results/v5_full/`)
2. `faithful_agg_edgeiiot_v5b_tolerance_attack.ipynb` (about 30 min; writes `results/v5b_tolerance/`)

Paths can be overridden with the environment variables `EDGE_DATA`, `CORE_DIR` and `OUT_DIR`.
All runs are deterministic given the seeds in `config.json`. The four plaintext baselines appear in
both experiments and must agree to the last digit; they do.

## Protocol summary (one round)

1. Servers train on the root set to obtain the reference update `r_t`; publish `B_t = kappa * ||r_t||` and `r_t / ||r_t||`.
2. Each site splits its update into two additive shares, commits to each share (Pedersen) and sends one share to each server.
3. Servers compute shares of `||u_i||^2` and `<u_i, r_t>^2` with Beaver triples; an ideal comparison reveals only the accept bit `b_i`.
4. Each server sums the accepted shares, publishes its aggregate share, the summed randomness, the accept bitmap and a Merkle root over its commitments.
5. Anyone checks `prod_{i accepted} C_i^j = Com(A_j; R_j)` for both servers; each site checks its Merkle inclusion path.

Real: sharing, fixed-point encoding, Beaver inner products, Merkle trees, Pedersen commitments and the audit.
Ideal: secure comparison (its communication is a placeholder estimate), range proofs, triple generation, secure clipping and in-aggregate noise.

## Headline results (Edge-IIoTset, 15 classes, 10 sites, 10 seeds)

| | clean | sign-flip | adaptive 30% |
|---|---|---|---|
| FedAvg | 0.706 | 0.091 | 0.046 |
| Multi-Krum (plaintext) | 0.690 | 0.686 | 0.693 |
| PVBRF-ID (tau = -0.3) | 0.708 | 0.696 | 0.571 |
| PVBRF-ID + secure clip | 0.697 | 0.688 | 0.641 |

Backdoors pass the filter (ASR 1.00); noise-based mitigation is heuristic (epsilon >= 16.9).
An update-inference attacker recovers class mix from plaintext updates (Spearman 0.81) but not
from one server's share (0.00). Communication is about 19x plain FedAvg upload.

## License

MIT, see `LICENSE`.
