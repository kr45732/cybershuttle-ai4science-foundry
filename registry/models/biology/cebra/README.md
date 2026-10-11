# CEBRA Achilles

I trained CEBRA-Behavior from scratch on `rat-hippocampus-single-achilles` and decoded position from the 3D embedding. No pretrained weights. The run is ICE job 6166173, one L40S, seed 0.

## Sources

CEBRA 0.6.1`: https://github.com/AdaptiveMotorControlLab/CEBRA

Paper: https://doi.org/10.1038/s41586-023-06031-6

Training settings: https://cebra.ai/docs/demo_notebooks/Demo_hippocampus.html

Decoder, 36 neighbors, cosine: https://cebra.ai/docs/demo_notebooks/Demo_decoding.html

Figure 2d: https://cebra.ai/docs/cebra-figures/figures/Figure2.html

Data is `cebra.datasets.init("rat-hippocampus-single-achilles")`: http://dx.doi.org/10.6080/K0862DC5

The loader downloads `achilles.jl.gz`, checks the MD5, and writes `data/rat_hippocampus/achilles.jl`. The scripts set `CEBRA_DATADIR` to `data/` before importing cebra. Parameters are in `config.yaml`.

## Setup

Copy this folder to scratch, connect to GlobalProtect, then ssh in. The login node is Python 3.9. CEBRA 0.6.1 needs 3.10+, so load `python/3.12.5`.

From the repo root:

```bash
rsync -av registry/models/biology/cebra/ GT_USERNAME@login-ice.pace.gatech.edu:scratch/cebra-achilles/
ssh GT_USERNAME@login-ice.pace.gatech.edu
module load python/3.12.5
cd ~/scratch/cebra-achilles
python3 -m venv ~/scratch/cebra-venv
source ~/scratch/cebra-venv/bin/activate
python -m pip install -r requirements.txt
```

The venv is on scratch because the torch wheel is large. `requirements.txt` pins cebra 0.6.1, pandas 3.0.6, and PyYAML 6.0.3. `requirements-freeze.txt` is the freeze from job 6166173, including `torch==2.14.1`.

## Run

From `~/scratch/cebra-achilles`:

```bash
sbatch scripts/ice_job.sbatch
squeue -u $USER
```

The job requests account `coc`, qos `coc-ice`, partition `ice-gpu`, one L40S, 8 cpus, 32G, and 2 hours. It runs `scripts/smoke_test.py` for 5 iterations, then `scripts/run_hippocampus.py` for 5000 iterations and a second fit with the training labels shuffled. Output goes in `results/`.

## Results

`logs/cebra-achilles-6166173.out` has `SMOKE PASS` and `RUN PASS` on an L40S, cuda True. Scores are in `results/metrics.json`.

- behavior: R² 0.801, median error 7.63 cm, fit 28 s
- shuffled labels: R² -0.021, median error 47.7 cm, fit 25 s
- Figure 2d: about 0.734 and 5.8 cm, shuffled about -0.499 and 44.7 cm

The behavior embedding tracks position. The shuffled one does not, so the decoder is using the labels. Figure 2d is a median of 10 runs and 3 splits, and this is one seed, so the R² values land further from the paper than the errors do. Embeddings are the `.npy` files. Predictions are the `.npz` files.

## Problems

A V100 job, 6166048, failed the smoke test with `no kernel image is available for execution on the device`. This torch build has no V100 kernel. The error is in `logs/cebra-achilles-6166048.err`. The sbatch file asks for an L40S.

If the Achilles dataset is `not found in options`, pandas is missing. CEBRA imports Allen before hippocampus, and Allen needs pandas.

If the job cannot find `~/scratch/cebra-venv`, create it on the login node before `sbatch`.

## Cleanup

```bash
scancel JOBID
rm -rf "$HOME/scratch/cebra-venv" "$HOME/scratch/cebra-achilles"
```
