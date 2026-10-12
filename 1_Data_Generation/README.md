# Part 1 — data10 generation

This directory contains only the code directly used to create and finalize the
data10 HR/LR ultrasound pairs. Old copies, debug scripts, execution logs,
W&B caches, temporary outputs, and model-training files were intentionally
removed.

## Role in the end-to-end workflow

Start with the repository-level `README.md` for the complete execution order.
This Part 1 implementation accepts an assembled
`Dataset_BUSI_with_GT/category/category (N).png` plus matching
`category (N)_mask.png` source tree and writes the handoff consumed by Part 2:

```text
WORK_ROOT/final_training_data_hr_derived_lr_shared_post_db40_gain_diff120/
├── sc1.0/scans/scan_CATEGORY_N_HR.png and .pkl
├── sc0.5/scans/scan_CATEGORY_N_LR.png and .pkl
└── scan_CATEGORY_N_HR_minus_LR_pixelwise.png
```

The `.pkl` files are required for Part 2's `final-db40` preparation mode;
they retain the generated image data and seed metadata. Run the one-sample
command in the root README before generating a batch. For an exact data10-sized
training set, complete at least 50 HR/LR pairs and any desired additional
LR-only cases before proceeding to Part 2.

## Source BUSI images and masks

The original BUSI (Breast Ultrasound Images) dataset is **not included in this
GitHub repository**. Part 1 currently contains no BUSI PNG, JPG, or mask image;
only code and a generated-file inventory are included. Keeping the raw dataset
out of Git avoids a large repository and avoids redistributing the medical
image archive inside this code package.

- BUSI original article and supplementary dataset:
  [Al-Dhabyani et al., *Dataset of breast ultrasound images*, Data in Brief 28 (2020), 104863](https://www.sciencedirect.com/science/article/pii/S2352340919312181)
- BUSI DOI: [10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863)
- BUSI official dataset information page:
  [Cairo University — Breast Ultrasound Images Dataset (Dataset BUSI)](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
- Expanded image/mask reference for local IDs beyond the public BUSI range:
  [Zenodo — *Towards Generalizable Ultrasound Segmentation via Physics-Guided Supervision*](https://zenodo.org/records/18449815)
  (`images.zip`)

Download and extract the datasets outside this Git repository. The directory
passed to `--busi-dataset-path` must be the folder named
`Dataset_BUSI_with_GT`, not its parent ZIP/download directory. If reproducing
the locally expanded IDs, place each matching Zenodo image/mask pair in its
appropriate category folder while retaining the `category (N).png` and
`category (N)_mask.png` filenames.

Expected layout:

```text
Dataset_BUSI_with_GT/
├─ benign/
│  ├─ benign (1).png
│  ├─ benign (1)_mask.png
│  └─ ...
├─ malignant/
│  ├─ malignant (1).png
│  ├─ malignant (1)_mask.png
│  └─ ...
└─ normal/
   ├─ normal (1).png
   ├─ normal (1)_mask.png
   └─ ...
```

For the current preprocessor, both `category (N).png` and its matching
`category (N)_mask.png` must exist. An unpaired image or mask is skipped.
When a BUSI case has additional files such as `_mask_1.png`, the current code
uses the primary `_mask.png` only.

### Important reproducibility distinction

The public BUSI archive is sufficient to run this generation pipeline for its
published sample IDs. It is **not sufficient to recreate every file in the
existing data10 dataset byte-for-byte**. The data10 inventory includes many
locally added/renumbered benign masks (for example IDs in the 4000–8000 range)
that are not part of the original public BUSI release. The corresponding
expanded image/mask collection is referenced above through Zenodo, but is not
stored in this Git repository. Exact data10 reproduction requires the same
expanded local mask collection and filenames; verify the downloaded contents
against the local data inventory before claiming an exact file-level match.

## Installation

The recorded environment used Python 3.10 and `k-Wave-python==0.6.2`. A clean
environment can be prepared as follows:

```powershell
conda create -n semfar-data python=3.10 -y
conda activate semfar-data
pip install -r requirements.txt
```

GPU execution additionally requires an NVIDIA GPU and a working k-Wave/CUDA
runtime. Test one case before starting a large batch. W&B is optional; log in
with `wandb login`, or add `--no-wandb` to run without it.

## First run from downloaded BUSI data

The following command preprocesses the downloaded BUSI images/masks, generates
one HR/LR pair with matched seeds, runs GPU k-Wave, and records the run in W&B:

```powershell
cd D:\path\to\SeMFAR_Github\1_Data_Generation
python generate_hr_lr_pairs.py `
  --busi-dataset-path "D:\datasets\Dataset_BUSI_with_GT" `
  --work-root "D:\semfar_data_generation" `
  --prepare-data `
  --sample-name benign_1 `
  --gpu `
  --wandb-project semfar-data-generation
```

`--prepare-data` scans and preprocesses the available BUSI pairs into
`WORK_ROOT/processed_phantoms`. It may process the full downloaded archive even
when only one simulation sample is requested. It is needed on the first run or
after changing the source masks. On later runs, omit it and reuse the processed
phantoms:

```powershell
python generate_hr_lr_pairs.py `
  --busi-dataset-path "D:\datasets\Dataset_BUSI_with_GT" `
  --work-root "D:\semfar_data_generation" `
  --sample-names benign_2 benign_3 benign_4 `
  --gpu `
  --wandb-project semfar-data-generation
```

Use explicit `--sample-name` or `--sample-names` when referring to original
BUSI IDs. `--sample-range` selects positions from the sorted processed list,
not necessarily the numeric IDs printed in the BUSI filenames.

Path options may also be set once per shell:

```powershell
$env:BUSI_DATASET_PATH = "D:\datasets\Dataset_BUSI_with_GT"
$env:SEMFAR_WORK_ROOT = "D:\semfar_data_generation"
python generate_hr_lr_pairs.py --sample-name benign_1 --gpu
```

Large local folders are excluded by the repository `.gitignore`; do not force
add `Dataset_BUSI_with_GT`, `local_data`, or `local_outputs` to GitHub.

## Output locations

With `--work-root D:\semfar_data_generation`, the main directories are:

```text
D:\semfar_data_generation\
├─ processed_phantoms\
├─ python_simulation_results_hr_derived_lr\
├─ final_training_data_hr_derived_lr\
├─ final_training_data_hr_derived_lr_shared_post_db40_gain\
└─ final_training_data_hr_derived_lr_shared_post_db40_gain_diff120\
```

The final directory contains the processed HR/LR PNG/PKL pair and its
pixel-wise comparison plot.

## Continue to Part 2 training

Part 1 deliberately keeps HR and LR in separate scale directories. After the
requested simulations finish, use
`../2_Model_Training_SeMFAR/prepare_training_data.py` to apply the final
data10-compatible DB40/fixed-gain transform, collect 50 complete HR/LR pairs
into one flat `Paired` folder, collect the remaining LR images into
`Unpaired`, and create the fixed 30/10/10 split. See the Part 2 README for the
complete commands. Do not point the training scripts directly at the nested
Part 1 output directories.

## Core workflow

1. `core/prepare_phantom.py` converts each matched BUSI image/mask pair into a
   256×256 simulation phantom.
2. `core/simulate_kwave.py` creates the HR acoustic map and runs k-Wave.
3. `core/simulate_hr_derived_lr.py` derives the LR acoustic map from HR using
   area downsampling, without independent re-standardization, and reuses the
   HR run/sample/noise seeds.
4. `core/postprocess_bmode.py` reconstructs B-mode output.
5. `postprocessing/shared_pair_normalization.py` and
   `postprocessing/fixed_pair_brightness.py` implement the paired display path.
6. The final data10 display transform is
   `postprocessing/final_db40_pair_reprocess.py`: HR and LR are each normalized
   with their own 99.9th percentile, compressed to 40 dB, and multiplied by the
   fixed gain `0.567379057497988`.
7. `visualization/plot_pair_difference_tumor_roi.py` creates the final
   pixel-wise comparison plot with tumor-only Pearson correlation.

## Seed control

Use `--run-seed` for one fixed run seed or `--seed-map` for a JSON mapping of
sample name to run seed. HR and LR reuse the paired seed information, and seed
metadata are saved in the generated PKL files.

```powershell
python generate_hr_lr_pairs.py `
  --work-root "D:\semfar_data_generation" `
  --sample-name benign_1 `
  --run-seed 123456789 `
  --gpu
```

## Reprocess an existing data10 pair

The batch script defaults to `D:\Woojeong_SR\data10\Paired_ver10`:

```powershell
.\run_final_db40_reprocess.ps1
```

Or supply paths explicitly:

```powershell
.\run_final_db40_reprocess.ps1 `
  -DatasetRoot "D:\path\to\Paired_ver10" `
  -Python "python"
```

Each completed paired sample contains these five files:

- `scan_benign_N_HR.pkl`
- `scan_benign_N_HR.png`
- `scan_benign_N_LR.pkl`
- `scan_benign_N_LR.png`
- `scan_benign_N_HR_minus_LR_pixelwise.png`

## File guide

| File | Role |
|---|---|
| `generate_hr_lr_pairs.py` | User-facing HR/LR generation entry point |
| `pipeline_runner.py` | Pipeline orchestration, sample selection, W&B and seed handling |
| `core/prepare_phantom.py` | BUSI image/mask and phantom preprocessing |
| `core/simulate_kwave.py` | Base k-Wave simulation |
| `core/simulate_hr_derived_lr.py` | HR-derived LR acoustic map and paired simulation |
| `core/postprocess_bmode.py` | RF/B-mode reconstruction |
| `config/pipeline_options.py` | Simulation and postprocessing configuration |
| `utils/file_io.py` | File and metadata helpers |
| `postprocessing/shared_pair_normalization.py` | Shared-pair 40 dB transform |
| `postprocessing/fixed_pair_brightness.py` | Pair brightness/difference scaling |
| `postprocessing/final_db40_pair_reprocess.py` | Authoritative final data10 transform |
| `visualization/plot_pair_difference.py` | Standard pair difference plot |
| `visualization/plot_pair_difference_tumor_roi.py` | Final tumor-ROI comparison plot |
| `metadata/data10_file_inventory.csv` | Inventory of generated data10 files; not raw images |
| `metadata/python_environment.txt` | Captured Python environment |
| `metadata/FILES_SHA256.csv` | Checksums for this cleaned Part 1 package |

## Reproduce the exact data10 source-mask subset

The raw medical-image archives are not distributed with this repository.
`metadata/data10_mask_manifest.csv` records the **1,038** mask files used by
data10: 530 from the public BUSI archive and 508 from the expanded Zenodo
collection. Every entry includes a SHA-256 checksum.

1. Download and extract the BUSI archive so that its root contains directories
   such as `benign/benign (1)_mask.png`.
2. Download and extract the Zenodo `images.zip`, and set `--zenodo-root` to
   the directory containing the expanded `benign (N)_mask.png` files.
3. Run the selector below. It checks every source mask against the recorded
   SHA-256 value and copies only the required files. A missing or different
   source file causes a non-zero exit code and is reported; it is never silently
   substituted.

```powershell
python prepare_data10_masks.py `
  --busi-root "D:\datasets\Dataset_BUSI_with_GT" `
  --zenodo-root "D:\datasets\phyusnet_images\images_prepare2" `
  --output-root "D:\semfar_data_generation\data10_source_masks"
```

Use `--dry-run` first to validate the downloaded source directories without
copying. The output preserves the category subfolder and original filename.
This script selects masks only; the simulation preprocessor also needs the
matching original image for each selected mask.

### Step-by-step: prepare exactly the data10 mask subset

This is the practical procedure for a new machine. It does **not** download or
redistribute any source medical image itself; you download the two source
archives from their original providers and the script selects the recorded
subset locally.

1. Download the original BUSI archive from the [Cairo University BUSI page](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset), then extract it. The path passed as `--busi-root` must directly contain `benign`, `malignant`, and `normal` folders.
2. Download `images.zip` from [Zenodo: *Towards Generalizable Ultrasound Segmentation via Physics-Guided Supervision*](https://zenodo.org/records/18449815), extract it, and identify the folder that directly contains expanded files such as `benign (6487)_mask.png`. Pass that folder as `--zenodo-root`.
3. From this `1_Data_Generation` folder, validate both downloads without writing any files:

```powershell
python prepare_data10_masks.py `
  --busi-root "D:\datasets\Dataset_BUSI_with_GT" `
  --zenodo-root "D:\datasets\phyusnet_images\images_prepare2" `
  --output-root "D:\semfar_data_generation\data10_source_masks" `
  --dry-run
```

4. A successful check reports `manifest_entries: 1038`, with empty `missing`
   and `hash_mismatch` lists. If either list is non-empty, do not continue:
   correct the source path or download the matching source archive again.
5. Run the same command again without `--dry-run`. It copies only the
   requested masks, retaining their original folder/file names:

```text
D:\semfar_data_generation\data10_source_masks\
└── benign\
    ├── benign (109)_mask.png
    ├── benign (4123)_mask.png
    └── ... 1,038 verified masks total
```

`metadata/data10_mask_manifest.csv` is the authoritative list. Its columns
mean: `sample_id` = pipeline sample name, `source` = `busi` or
`zenodo_expanded`, `relative_path` = expected location below that source root,
and `sha256` = exact-content check. The list contains 530 BUSI masks and 508
expanded Zenodo masks.

> **Important:** the k-Wave preprocessor needs a matching original image as
> well as a mask. This selector intentionally handles masks only, because it
> documents the source subset without redistributing source data. Before
> running `generate_hr_lr_pairs.py --prepare-data`, ensure that each selected
> `category (N)_mask.png` has its corresponding `category (N).png` available
> in the dataset structure expected by `--busi-dataset-path`.

