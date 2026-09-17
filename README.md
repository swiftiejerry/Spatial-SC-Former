# Spatial-scFormer

A test of whether two common fixes recover rare tissue domains that a baseline
model misses: giving a spatial transcriptomics model the tissue coordinates it
was not using, and letting its cluster targets move during training.

They do not. This repository holds the code, the run outputs, and a [15-page
report](report/The_Lock_Is_Not_the_Limit.pdf) explaining why.

## What the question was

scFormer (Huang et al., *Advanced Biotechnology* 2026) learns cell and gene
embeddings on a heterogeneous graph and clusters them. It generates its cluster
targets once, before training, using Leiden, then freezes them and trains the
network to reproduce them. Two problems follow from that design. The model has no
way to undo a mistake the initial clustering made, and it never sees where a spot
sits in the tissue.

The obvious remedies are to add a spot-to-spot spatial relation to the graph, and
to replace the frozen target with prototypes that update during training. We
implemented both and ran them on one human lymph node section (3,484 spots,
18,085 genes, 10 annotated domains, 5 of them rare).

## What happened

Neither remedy recovered the rare domains. The pre-registered endpoint,
rare-domain F1, did not move in either direction.

The reason is more interesting than the outcome. The model trained against the
frozen Leiden target reproduces it almost exactly: at one seed it changed **0 of
3,484** assignments over 100 epochs. So the first round of comparisons measured
noise, not the effect of the spatial relation. That is the lock.

But unlocking it does not help. On a rebuilt evaluation where the output can
actually move (62%, 51%, 60% of spots differ from the pseudo-label, against
0–1.9% before), the unweighted spatial relation made structural agreement worse
on all three seeds:

| metric | B0 − A* (3 seeds) | |
|---|---|---|
| ARI | −0.1177 ± 0.0585 | same sign on every seed |
| NMI | −0.1411 ± 0.0806 | same sign on every seed |
| boundary-F1 | −0.0399 ± 0.0141 | same sign on every seed |
| rare-domain F1 (primary) | +0.0592 ± 0.0986 | no consistent direction |

At three seeds only boundary-F1 reaches p < 0.05 (ARI p = 0.073, NMI p = 0.094),
so we report the paired tests alongside the effect sizes rather than claiming
significance. The spatial edges were verified live: messages and gradients are
non-zero at every layer, and removing them reproduces the baseline to within
1e-6. The relation reached the model, and it made the representation worse.

Prototype clustering failed the same way. Given an initial partition that does
contain a rare cluster, the update erased it: rare-domain F1 fell from 0.3927 to
0.0619, and the cluster fragmented into pieces that are *below* the section-wide
rare rate.

## Three things that are true independently of the negative result

A rare-domain F1 of zero does not mean the cluster count is too low. We enumerated
all 42,525 ways of merging the 10 annotated domains into 5 groups; the best
attainable rare-domain F1 is 1.0000. What actually pins the endpoint at zero is
whether the initial partition happens to contain a cluster below the rarity
threshold, which is a property of the partition and not of the model. The construction
merges ground-truth labels, so it bounds what is *achievable*, not what an
unsupervised pipeline will find.

A spatial relation is untestable unless the batching keeps its edges. The
upstream sampler scatters a batch of 30 spots across the whole section, so at
k=10 only 0.78% of spatial edges have both endpoints in the same batch. Spatial
median bisection raises that to 69.4%, and a one-hop halo to 100%, at the cost of
growing the batch to 58–81 spots. Without this fix, an arm comparison measures
the sampler.

The frozen target leaves almost no room to measure anything. With the
resolution-0.8 partition, the baseline arm deviated from its pseudo-label by
0, 1, and 65 spots across seeds, against 7, 0, and 5 for the spatial arm. Which
arm deviates, and by how much, is decided by the seed rather than the relation.
A paired difference inside that headroom can only be reported as "not
measurable", never as "no effect".

## Method details worth knowing before reading the numbers

Only one of the five rare domains is recoverable from expression alone. A
Wilcoxon screen over the full gene ranking finds a specific, statistically
sound marker only for follicle (CXCL13, detected in 89.6% of the domain, 10.1×
the section mean, adjusted p = 3e-41). Hilum's COL1A2 passes the statistical
criteria but is not spatially specific: collagen is high across the whole
capsule, so the field does not localize the domain. The remaining three domains
fail at least one criterion: medulla vessels and subcapsular sinus have no gene
meeting all three, and trabeculae, with 8 spots, has no gene surviving
multiple-testing correction (its top gene has adjusted p = 1.0). For those
domains, the premise that a signal exists and the model failed to use it has not
been established.

Two reading rules we apply to our own numbers. The 30-epoch aligned comparison
and the 100-epoch runs are separate measurements and their absolute values do
not compare. And a degree-preserving rewiring control, which shows the trained
checkpoint failing to generalize to a random graph, runs at inference time on a
model trained with the true graph. Our own protocol labels it "diagnostic
necessity evidence only, not retrained causal null", so it bounds what the
trained model does, not what the pipeline extracts from spatial topology.

## Repository layout

```
report/
  The_Lock_Is_Not_the_Limit.pdf   final report, 15 pages
  source/                         LaTeX source + 12 vector figures (compiles with pdflatex)
src/spatial_scformer/             method code
  graph/                          spatial kNN graph, spatial batching + halo, Top-Z edge selection
  cluster/dynamic.py              prototype clustering: confidence-gated refresh, reseeding, fixed K
  model/hgt.py                    heterogeneous graph model (spot/gene nodes + spatial relation)
  losses.py                       L_KL and L_proto
  stage1_trainer.py stage2_trainer.py train.py
scripts/
  run_stage0_baseline.py          upstream scFormer baseline
  run_stage1.py                   arm A/B: spatial relation on or off, nothing else changes
  run_stage23.py                  arm C/D: dynamic clustering with and without spatial
  figures/                        scripts that regenerate the report figures
configs/                          run configurations and protocol snapshots
experiments/
  results.json                    one row per run, merged from the per-run metrics files
  recompute_churn.py              recomputes the churn numbers quoted in this README and the report
  runs/                           per-run outputs (metrics, initial and final partitions)
  reports/                        aggregate reports the README and the report cite
    goal_stage1/ goal_stage2/     the A/B and C arms
    fixed_pseudolabel_headroom/   how far each arm can deviate from its target
    rare_domain_baseline_frontier/  rare-domain F1 across Leiden resolutions
    rare_f1_ceiling/              the 42,525-partition construction proof
    marker_screen/                per-domain Wilcoxon statistics
    advanced_blueprint_P0..P2/    the aligned-basis runs; P2 is the paired comparison that stopped the project
third_party/scFormer/             upstream snapshot (commit 7401620), with explanatory comments added
docs/data.md                      how to obtain and place the dataset
```

## Running it

```bash
pip install -r requirements.txt
```

The dataset is not distributed here. Place `adata_RNA.h5ad` in
`data/raw/human_lymph_node_A1/`; the requirements are in
[docs/data.md](docs/data.md). A GPU is optional; the full runs fit in about 1.6 GB.

```bash
# Stage 0: upstream scFormer baseline
python scripts/run_stage0_baseline.py --epochs 100

# Stage 1: A vs B, differing only in the spatial relation switch
python scripts/run_stage1.py --arm A --seed 0
python scripts/run_stage1.py --arm B --seed 0 --spatial-k 10

# Stage 2/3: C vs D, dynamic clustering with and without spatial
python scripts/run_stage23.py --arm C --seed 0
python scripts/run_stage23.py --arm D --seed 0
```

Each run writes `metrics.json` and its partitions to `experiments/runs/<run_id>/`.
The run id encodes the switches, so protocols never overwrite each other.

To rebuild the report:

```bash
cd report/source && pdflatex main.tex && pdflatex main.tex && pdflatex main.tex
```

## What the results do and do not support

Supported: the unweighted k=10 spatial relation degrades structural metrics on
the aligned basis; prototype clustering erases a small cluster it is given; the
frozen target leaves thousandths of headroom; the partition, not the cluster
count, pins rare-domain F1 at zero on the default settings.

Not supported: that spatial relations are useless, or dynamic clustering is
useless. What was falsified is the specific configuration tested. Also not
supported: that the model cannot extract spatial topology. The rewiring control
is an inference-time swap, not a retrained null.

## License

Method base: scFormer (DOI 10.1007/s44307-026-00121-y). The upstream snapshot in
`third_party/` remains under its authors' copyright. New code here is MIT, see
[LICENSE](LICENSE).
