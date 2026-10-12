# Part 2 - training from newly generated Part 1 data

This folder preserves the archived 50-paired experiment evidence and adds a
portable workflow for training on a newly generated Part 1 dataset.

The runnable workflow covers:

- EMA semi-supervised EDSR
- SeMFAR affine residual modulation, including its internal upstream EDSR and
  residual-predictor dependencies

PFT remains results-only. Its exact original training implementation and
checkpoint provenance were not present in the archived package, so this
repository does not silently replace it with a different model.

## Role in the end-to-end workflow

Follow the repository-level `README.md` in order. This part does not run
k-Wave and does not consume raw BUSI files. Its required input is Part 1's
completed `final_training_data_hr_derived_lr_shared_post_db40_gain_diff120`
directory, containing both scale folders and HR/LR `.png` **and** `.pkl`
files. `prepare_training_data.py` converts that nested output into the flat
`Paired/` and `Unpaired/` directories plus `split.json` required by
`run_experiments.py`.

Before any 300-epoch job, run `run_experiments.py` with one seed and
`--epochs 1`. The runnable models are EMA and SeMFAR; SeMFAR internally trains
its upstream EDSR and residual predictor from the newly prepared paired data.

## 1. Generate the ultrasound data in Part 1

Follow `../1_Data_Generation/README.md`. A completed sample must have all four
files below:

```text
WORK_ROOT/final_training_data_hr_derived_lr_shared_post_db40_gain_diff120/
|- sc1.0/scans/scan_benign_N_HR.png
|- sc1.0/scans/scan_benign_N_HR.pkl
|- sc0.5/scans/scan_benign_N_LR.png
`- sc0.5/scans/scan_benign_N_LR.pkl
```

Part 1 creates nested HR and LR folders. The training scripts require flat
`Paired` and `Unpaired` folders, so run the preparation step below before
training.

## 2. Build training-ready Paired/Unpaired folders

From this directory:

```powershell
python prepare_training_data.py `
  --generation-root "D:\semfar_data_generation" `
  --part1-root "D:\path\to\SeMFAR_Github\1_Data_Generation" `
  --output-root "D:\semfar_training_data" `
  --paired-count 50 `
  --unpaired-count 0 `
  --selection-seed 2026 `
  --postprocess-mode final-db40
```

`final-db40` is the data10-compatible choice. For every selected image it
reconstructs from the generated PKL, applies per-image 99.9th-percentile
normalization, 40 dB compression, and the fixed gain
`0.567379057497988`. HR brightness never determines LR brightness, and LR
brightness never determines HR brightness.

The result is:

```text
D:\semfar_training_data\
|- Paired\
|  |- scan_benign_N_HR.png
|  `- scan_benign_N_LR.png
|- Unpaired\
|  `- scan_benign_M_LR.png
|- split.json              # one fixed 30/10/10 split
`- dataset_manifest.json   # sample lists and preparation settings
```

The preparation script verifies complete pairs and guarantees that Paired and
Unpaired sample IDs do not overlap. `--unpaired-count 0` uses every remaining
completed LR sample. To reproduce 50 paired plus 988 unpaired images exactly,
Part 1 must finish at least 1,038 distinct samples.

The public BUSI download contains fewer cases and does not include the local
4000-8000-series IDs used by the historical data10 experiment. It can run the
method and create a valid new experiment, but it cannot reproduce the exact
historical sample identities or the 50+988 count without the expanded local
mask collection.

For a quick file-wiring test only, `--postprocess-mode existing` copies the
Part 1 shared-pair PNGs. It does not apply the final data10 display transform.

## 3. Inspect the training commands without training

```powershell
python run_experiments.py `
  --data-root "D:\semfar_training_data" `
  --output-root "D:\semfar_new_runs" `
  --seeds 42 123 456 `
  --epochs 300 `
  --dry-run
```

This writes `run_plan.json` and prints every command in execution order.

## 4. Run on GPU

```powershell
python run_experiments.py `
  --data-root "D:\semfar_training_data" `
  --output-root "D:\semfar_new_runs" `
  --seeds 42 123 456 `
  --epochs 300
```

A real run stops immediately if CUDA is unavailable. `--allow-cpu` exists only
for an intentionally slow CPU test. The runner also executes a tiny CUDA kernel
before training, because `torch.cuda.is_available()` can be true even when the
installed PyTorch build does not support a newer GPU architecture.

Check the environment before a long run:

```powershell
python -c "import torch; x=torch.ones(1,device='cuda'); print(torch.__version__,torch.cuda.get_device_name(0),(x+1).item())"
```

This command must finish without a compute-capability or `no kernel image`
error.

### GPU compatibility requirement

`torch.cuda.is_available()` alone is not enough. The installed PyTorch CUDA
build must support the compute capability of the installed GPU. For example,
an RTX 5090 reports `sm_120`; a PyTorch build that supports only up to `sm_90`
will detect that GPU but cannot execute training kernels. In that situation,
install a current CUDA-enabled PyTorch build that explicitly supports your GPU
before running the 300-epoch command. Do not use `--allow-cpu` for the real
experiment; it exists only for a slow functional smoke test.

For an RTX 5090 on Windows, install the CUDA 12.8 build into the same Python
environment used for training, then install the remaining requirements:

```powershell
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The second command preserves the already installed CUDA build because it
satisfies `torch>=2.1`. It also installs `monai` and `einops`, both required by
the frozen EchoCare encoder used in SeMFAR.

For each seed, the runner performs:

1. EMA on the fixed Paired split and the new Unpaired LR folder.
2. A dedicated **SeMFAR upstream EDSR** training run on the new Paired split.
   It is an internal dependency of SeMFAR, not a separately reported Fully
   Supervised baseline.
3. A residual predictor trained on that seed's upstream EDSR checkpoint; its
   ridge constant is recalibrated from the current validation split rather
   than reused from data10.
4. The SeMFAR affine head using those newly trained upstream/residual
   checkpoints.

The upstream EDSR and residual predictor are therefore re-trained from the new
Paired data for every seed. EchoCare is an external pretrained encoder and is
kept frozen; it is not claimed to have been trained on these 50 pairs.

## 5. Files added for the runnable workflow

| File | Purpose |
|---|---|
| `prepare_training_data.py` | Final DB40 processing, flat folders, no-overlap checks, fixed split |
| `run_experiments.py` | GPU experiment runner for seeds 42/123/456 |
| `runnable/common.py` | Shared EDSR, residual-transformer, dataset, and metric utilities |
| `runnable/train_upstream_edsr.py` | Trains the internal upstream EDSR required by SeMFAR |
| `runnable/train_residual_predictor.py` | Trains the new-data residual predictor |
| `runnable/train_semfar.py` | Trains the SeMFAR affine residual head |
| `requirements.txt` | Part 2 Python dependencies |

The standalone Fully Supervised baseline is intentionally excluded from this
package. `05_Table3_Evaluation` is retained as an immutable historical Table 3
record and may therefore still contain its earlier multi-model comparison
entries, including Fully Supervised. New EMA and SeMFAR results are written
only to `--output-root`.
