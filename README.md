# Cloud Workload Forecasting Challenge

AI/ML take-home challenge built from the **Microsoft Azure Public Dataset V2** VM trace.

## Start here

1. Read [`ASSIGNMENT.md`](ASSIGNMENT.md).
2. Read [`DATASET.md`](DATASET.md).
3. Download the **`data-v1` GitHub Release** dataset and extract it as `challenge_data/`.
4. Install the project and analysis dependencies.
5. Run the analysis notebook and generate `predictions.csv`.

The challenge dataset is distributed as a versioned GitHub Release rather than committed to the repository. The `data-v1` release contains the fixed curated dataset used for this assignment.

## Environment setup

Python **3.11 or newer** is required.

Create a virtual environment:

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
```

Install the project, development, and notebook-analysis dependencies:

```bash
python -m pip install -e ".[dev,analysis]"
```

Run the tests:

```bash
python -m pytest -q
```

Execute the analysis notebook in Windows PowerShell without loading global
Jupyter configuration:

```powershell
$env:JUPYTER_CONFIG_DIR = "$PWD\.jupyter_empty"
New-Item -ItemType Directory -Force $env:JUPYTER_CONFIG_DIR | Out-Null
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks\forecasting_analysis.ipynb
```

Generate the required predictions from the fixed release:

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

The exploratory analysis, time-aware model comparison, final holdout evaluation,
and operational discussion are in [`notebooks/forecasting_analysis.ipynb`](notebooks/forecasting_analysis.ipynb).
The current data-backed results are summarized in [`reports/final_report.md`](reports/final_report.md).

## Dataset download

The hiring dataset is distributed as the fixed `data-v1` release of this repository.

**Download:** [Azure Dataset v1 Release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)

Download `data-v1.tar.gz` from the release and extract it into the repository as `challenge_data/`:

### Windows PowerShell

```powershell
curl.exe -L --fail -o data-v1.tar.gz https://github.com/atomityhq/forecasting-challenge/releases/download/data-v1/data-v1.tar.gz
New-Item -ItemType Directory -Force challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
```

Verify the published SHA-256 checksums:

```powershell
$root = (Resolve-Path challenge_data).Path
Get-Content (Join-Path $root "SHA256SUMS") | ForEach-Object {
    $expected, $name = $_ -split "\s+", 2
    $actual = (Get-FileHash (Join-Path $root $name) -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $expected) { throw "Checksum mismatch: $name" }
    "$name`: OK"
}
```

### Linux / macOS

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
cd challenge_data
sha256sum -c SHA256SUMS
```

You should see:

```text
vm_cpu_readings.csv.gz: OK
vm_metadata.csv.gz: OK
DATA_CARD.md: OK
```

The extracted directory should contain `vm_cpu_readings.csv.gz`,
`vm_metadata.csv.gz`, `DATA_CARD.md`, and `SHA256SUMS`.

The release is a real, curated subset of the Microsoft Azure Public Dataset V2. It is not synthetic data.

The dataset contains 10,000 VMs with 134 CPU measurements per VM at five-minute intervals, for a total of 1,340,000 CPU observations.

Use the exact `data-v1` release provided for this assignment. Do not substitute another dataset or a different version of the Azure Public Dataset.

## What you are solving

The underlying data contains anonymized VM workload telemetry sampled every five minutes. Your job is to understand the workload and build a defensible forecasting pipeline for near-term aggregate demand.

The challenge is intentionally open-ended. We evaluate reasoning, time-series methodology, leakage prevention, error analysis, reproducibility, and the quality of your operational conclusions—not just model complexity.

## Source

Microsoft Azure Public Dataset V2:
https://github.com/Azure/AzurePublicDataset

The official V2 documentation describes a 30-day VM trace with five-minute CPU readings, VM metadata, and encrypted subscription/deployment/VM identifiers. The full upstream trace is much larger than this challenge release.

## Submission

Submit your work as a repository or archive according to [`SUBMISSION.md`](SUBMISSION.md).

The pipeline constructs the two required aggregate targets from VM-level
readings, reserves the final 24 timestamps, and compares persistence, moving
average, linear trend, Ridge autoregression, and ordinary least-squares AR(12)
using expanding-window validation within the preceding training window. The
submitted CPU forecast is the AR(12) model fit on the first 110 observations;
this is the declared submission model, not necessarily the lowest-MAE
validation candidate. The public holdout has already been inspected during
development, so its metrics are descriptive and must not be presented as an
independent final test estimate.
