# Generative models for nondominated trajectories (fixed 4×4 grid)

Six notebooks, one per generative model, all on the same fixed grid and the same test. Each notebook is self-contained: upload it to Colab on its own and choose Run all. It needs no other file and no Drive access.

| Notebook | Model | Training per seed, Colab GPU | Whole notebook, 2-core CPU |
| --- | --- | --- | --- |
| `GridMO_gen_LSTM.ipynb` | Conditional LSTM, one move at a time | 9–17 s | about 4 min |
| `GridMO_gen_Transformer.ipynb` | Causal Transformer, Decision-Transformer style (told the score still to earn) | 16–24 s | about 7 min |
| `GridMO_gen_CVAE.ipynb` | Conditional VAE over the whole move sequence | 19–26 s | about 2 min |
| `GridMO_gen_Diffusion_MDLM.ipynb` | Masked discrete diffusion over move tokens (MDLM) | 19–26 s | about 9 min |
| `GridMO_gen_Diffusion_DecisionDiffuser.ipynb` | Diffusion over the path's positions, start and goal pinned (Diffuser / Decision Diffuser) | 19–27 s | about 7 min |
| `GridMO_gen_Diffusion_PCD.ipynb` | Pareto-conditioned diffusion over relaxed moves, with dominance reweighting (PCD) | 21–31 s, twice (with and without reweighting) | about 12 min |

Each notebook writes its per-seed numbers to `results/<model>.json`.

## The test (same in every notebook)

* **Grid:** start (4, 1), goal (1, 4), grass (2, 2), pool (3, 3), T = 10 steps. Scores Σ t·2^(−d_t) / Σ t for the grass and the pool, both higher is better.
* **Ground truth:** all 23,508 trajectories are listed; the exact front has 29 score pairs, shared by 240 trajectories.
* **The front is held out.** Models train on the 23,268 dominated trajectories (20,942 training, 2,326 validation), each labelled with its own scores, so a front hit is always something the model never saw. Copying training data scores zero.
* **Conditioning:** target scores (grass, pool). At test time each model gets 20 targets built from training data only, slightly beyond the best training scores in 20 directions, with 50 samples each: 1,000 trajectories per seed, 3 seeds.
* **Baselines:** random valid trajectories, and copying the training trajectories closest to each target.

## Results (Colab GPU, mean ± standard deviation over 3 seeds)

| Model | Valid | On the front (of valid) | Valid × on the front (of all) | Front points reached (of 29) | Distinct (of valid) | Target error |
| --- | --- | --- | --- | --- | --- | --- |
| Decision-Diffuser style, guidance 2 | 87.4% ± 0.7 | 53.7% ± 4.0 | **46.9%** | 27.7 ± 1.0 | 35.2% ± 3.1 | 0.029 ± 0.001 |
| LSTM, constrained decoding | 100.0% ± 0.0 | 39.6% ± 1.7 | 39.6% | 27.7 ± 1.2 | 39.9% ± 1.4 | 0.025 ± 0.001 |
| LSTM, unconstrained | 80.7% ± 8.4 | 45.0% ± 3.8 | 36.3% | 27.3 ± 1.2 | 39.4% ± 2.2 | 0.022 ± 0.001 |
| MDLM, guidance 2 | 73.5% ± 0.9 | 24.3% ± 1.9 | 17.9% | 24.7 ± 0.5 | 54.6% ± 4.4 | 0.058 ± 0.004 |
| Transformer, constrained decoding | 100.0% ± 0.0 | 17.5% ± 4.2 | 17.5% | 26.3 ± 0.5 | 57.9% ± 3.5 | 0.035 ± 0.002 |
| Transformer, unconstrained | 55.7% ± 5.8 | 24.9% ± 3.2 | 13.9% | 24.3 ± 1.0 | 56.8% ± 5.1 | 0.023 ± 0.002 |
| PCD, shared targets | 37.6% ± 8.5 | 27.6% ± 4.8 | 10.4% | 22.0 ± 0.8 | 72.9% ± 7.2 | 0.068 ± 0.006 |
| PCD, its own targets | 37.7% ± 8.1 | 25.3% ± 2.1 | 9.5% | 20.7 ± 1.2 | 73.6% ± 6.3 | 0.067 ± 0.005 |
| Same diffusion, no reweighting | 38.6% ± 12.6 | 24.2% ± 4.9 | 9.3% | 23.7 ± 3.3 | 75.3% ± 12.1 | 0.072 ± 0.010 |
| CVAE, constrained decoding | 100.0% ± 0.0 | 6.8% ± 0.3 | 6.8% | 17.7 ± 1.7 | 64.4% ± 3.3 | 0.160 ± 0.003 |
| CVAE, unconstrained | 13.1% ± 1.7 | 30.0% ± 4.4 | 3.9% | 13.3 ± 1.9 | 71.3% ± 2.9 | 0.059 ± 0.005 |
| Random valid trajectories | 100.0% ± 0.0 | 1.2% ± 0.1 | 1.2% | 8.7 ± 1.9 | 97.9% ± 0.4 | – |
| Copy closest training trajectories | 100.0% ± 0.0 | 0.0% ± 0.0 | 0.0% | 0.0 ± 0.0 | 42.5% ± 0.0 | 0.027 ± 0.000 |

"Valid × on the front (of all)" is the share of all samples that are usable front trajectories, the fairest single number across models because some models produce invalid paths. Here it is the product of the two mean columns; the per-seed values are in `results/*.json`. "Constrained decoding" blocks moves that leave the grid or can no longer reach the goal; only models that write moves in order can use it.

## What the results say

* **Every model generates front trajectories it never saw.** They reach 13 to 28 of the 29 front points; random sampling reaches 8.7, and copying training data reaches none.
* **Decision-Diffuser-style diffusion is best**, on both the GPU (46.9% usable) and the CPU run (45.5%), and was steadier across seeds on the GPU. It repeats itself: only 35% of its samples are distinct.
* **The LSTM is a close second and the most dependable:** 39.6% usable, every sample valid, and as many front points reached as the Decision-Diffuser model (27.7 of 29).
* **PCD's reweighting did not hold up.** On the GPU, PCD with reweighting gives 10.4% usable against 9.3% without it, and the same validity (38% vs 39%). The CPU run's gain (22.7% vs 8.9%) was within run-to-run variation, so the effect of reweighting is unresolved.
* **The middle of the ranking is not reliable.** MDLM (17.9%), the Transformer (17.5%) and PCD (10.4%) changed order between CPU and GPU; differences of a few points between them mean little with 3 seeds.
* **Models that choose all moves at once break paths.** Invalid samples: Decision Diffuser 13%, MDLM 27%, PCD 62%, CVAE 87% unconstrained. The LSTM and Transformer stay 100% valid by blocking illegal moves as they write.
* **Conditioning on the target is what drives it.** With guidance 0 (target ignored), the diffusion models barely hit the front (0.1–3.1% of valid samples, against 1.2% for random sampling). Guidance 1 to 2.5 works best; at 4 the Decision-Diffuser model's validity fell to 20% on the GPU (90% on the CPU), so strong guidance is unstable.
* **The shared targets are not the bottleneck.** Asked for the exact front scores, models hit the front less often than with the shared targets (LSTM 27% vs 40%).

## Same notebooks, CPU and GPU

Same code and same seeds; the GPU draws different random numbers, so the models train and sample differently. The baselines, which do not depend on PyTorch, come out identical, which confirms the scoring itself is deterministic.

| Model | Usable, CPU | Usable, GPU | Change | Front points, CPU | Front points, GPU |
| --- | --- | --- | --- | --- | --- |
| Decision-Diffuser style | 45.5% | 46.9% | +1.4 | 27.3 | 27.7 |
| LSTM, constrained | 34.2% | 39.6% | +5.4 | 28.3 | 27.7 |
| MDLM | 21.5% | 17.9% | −3.6 | 23.7 | 24.7 |
| Transformer, constrained | 20.2% | 17.5% | −2.7 | 27.3 | 26.3 |
| PCD, shared targets | 22.7% | 10.4% | −12.3 | 23.3 | 22.0 |
| Same diffusion, no reweighting | 8.9% | 9.3% | +0.4 | 24.3 | 23.7 |
| CVAE, constrained | 7.7% | 6.8% | −0.9 | 18.3 | 17.7 |
| Random valid trajectories | 1.2% | 1.2% | 0.0 | 8.7 | 8.7 |

The top two (Decision Diffuser, LSTM) and the bottom (CVAE) agree on both machines. PCD moved by 12 points, more than its spread across seeds, so 3 seeds understate run-to-run variation.

## Caveats

* One small grid. Models only have to learn about 23,000 short sequences, so this shows the protocol works; it does not yet show which model scales. The same notebooks run on other settings by changing `CONFIG` (keep T ≤ 12, since every trajectory is listed).
* Three seeds are too few to rank the middle group; use at least 10 before drawing conclusions about MDLM, the Transformer or PCD.
* Equal, small training budgets and no per-model tuning; the Transformer's and MDLM's validation losses were still falling.
* PCD's bin count, K and τ are our choices; the paper gives them in an appendix we could not read.
* The hypervolume ratio is near 1 even for the baselines, because the best dominated trajectories sit right next to the front. The notebooks report it, but the comparison above leaves it out.

## References

* LSTM: Hochreiter and Schmidhuber, 1997. Decision Transformer: [Chen et al., NeurIPS 2021](https://arxiv.org/abs/2106.01345). Conditional VAE: [Sohn et al., NeurIPS 2015](https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html).
* MDLM: [Sahoo et al., NeurIPS 2024](https://arxiv.org/abs/2406.07524); D3PM: [Austin et al., NeurIPS 2021](https://arxiv.org/abs/2107.03006); classifier-free guidance: [Ho and Salimans, 2022](https://arxiv.org/abs/2207.12598).
* Diffuser: [Janner et al., ICML 2022](https://arxiv.org/abs/2205.09991); Decision Diffuser: [Ajay et al., ICLR 2023](https://arxiv.org/abs/2211.15657).
* PCD: [Shrestha et al., 2026](https://arxiv.org/abs/2602.00737). Runner-up not built: MODULI ([Yuan et al., ICML 2025](https://icml.cc/virtual/2025/poster/45086)), which conditions on preference weights.
