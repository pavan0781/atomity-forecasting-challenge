# Cloud Workload Forecasting: Results

## Executive summary

The fixed `data-v1` release contains a complete, regular panel of 10,000 VMs
observed every five minutes for 134 points. Every VM has a valid observation at
every timestamp, so `active_vm_count` is exactly 10,000 throughout this curated
trace. This target is required, but it provides no evidence about VM churn in
this sample.

The submitted CPU forecast is an ordinary least-squares AR(12): an
autoregression with an intercept and 12 lags, fit on the first 110 observations
and recursively forecast for 24 steps. It was retained as the requested
submission method, not because it won validation. Across four expanding-window
validation folds, AR(12) MAE was 0.2446 percentage points, compared with 0.2056
for persistence and 0.2008 for six-lag Ridge. On the public 24-point holdout,
AR(12) MAE was 0.0986 and persistence MAE was 0.1468.

The public holdout has already been inspected during development, including
multiple forecast iterations. Its score is therefore descriptive for this one
window, not an independent or unbiased final test estimate. Validation favors
Ridge or persistence over AR(12), and AR(12) validation error on high-load
observations is worse than persistence. Use the forecast as a planning signal
with conservative headroom and monitoring, not as a standalone automated
capacity control.

## Data and target construction

The release contains 1,340,000 CPU readings from 10,000 anonymized VMs and 134
timestamps, from source second 0 to 39,900 at 300-second intervals. Validation
found no missing values, duplicate `(timestamp, vm_id)` pairs, irregular
timestamp steps, or average CPU readings outside `[0, 100]`. Every VM had
exactly 134 readings and a corresponding metadata row.

At each timestamp:

- `active_vm_count` counts distinct VMs with a valid `avg_cpu` reading.
- `aggregate_cpu_utilization` is the unweighted arithmetic mean of valid
  VM-level `avg_cpu`, in percentage points.

No target values were imputed. All target timestamps were present. The first
110 observations (source seconds 0–32,700) formed the modeling window; the last
24 (33,000–39,900) were reserved as the two-hour holdout.

The metadata CPU summary columns were not used as predictors because they
summarize workload behavior beyond an individual forecast origin. Metadata
segments were used only for retrospective workload description.

## Workload profile

Across all 134 timestamps, mean aggregate CPU utilization was 9.388%, with
standard deviation 0.195 percentage points, a minimum of 8.984%, and a maximum
of 9.912%. In the first 110 observations, CPU ranged from 9.112% to 9.912%;
the training mean was 9.414%.

The category-level CPU distribution differed materially in its tails:

| VM category | VMs | Mean VM CPU (%) | Median VM CPU (%) | 95th percentile (%) |
|---|---:|---:|---:|---:|
| Delay-insensitive | 6,180 | 8.236 | 5.150 | 25.913 |
| Interactive | 3,091 | 11.493 | 5.278 | 49.412 |
| Unknown | 729 | 10.233 | 3.378 | 52.229 |

Interactive and unknown workloads have much higher 95th-percentile utilization
than their medians. A single fleet-wide mean can conceal high-utilization
subpopulations; production planning should monitor workload-specific tails as
well as the aggregate.

An Augmented Dickey–Fuller test on the 110-point training CPU series returned
`p = 0.413`. This is insufficient evidence to reject a unit-root null, but the
short series makes the test low-powered and it does not establish whether the
series is stationary. The trace is too short to support a daily seasonal model.

## Forecasting and validation

Candidates were persistence, recent-window means (6, 12, or 24 observations),
linear trends over those windows, Ridge autoregressions over 6, 12, or 24 lags,
and ordinary least-squares AR(12). Ridge standardizes lag features using only
each fold's training data, fits a ridge-penalized linear autoregression
(`alpha = 1`), and forecasts recursively. AR(12) fits an intercept and 12 lag
coefficients by least squares, then forecasts recursively.

All candidates used the same expanding chronological folds: training origins
at observations 50, 62, 74, and 86, each evaluated over the following 24
observations. The horizons overlap, so fold errors are correlated and should
not be interpreted as independent replicates. Model comparison minimized
mean fold MAE; the submitted AR(12) was retained as the requested method, not
selected because it won validation.

| Model | Mean fold MAE (percentage points) | Mean fold RMSE | High-load MAE* |
|---|---:|---:|---:|
| Ridge AR, 6 lags (lowest MAE) | 0.2008 | 0.2282 | 0.2433 |
| Persistence | 0.2056 | 0.2387 | 0.1896 |
| OLS AR(12), submitted | 0.2446 | 0.2769 | 0.2485 |

AR(12) validation MAE is about 18.9% higher than persistence and 21.8% higher
than the lowest-MAE Ridge candidate. Its high-load MAE is also worse than
persistence. The high-load threshold is the 90th percentile of the training
values in each fold; model comparison did not use holdout labels. These results
caution against treating the submitted method as validation-optimal.

The active-VM series is constant at 10,000 for the entire dataset. Its
validation and holdout errors are zero for constant forecasts, including
persistence. This is a consequence of the release's complete-panel
construction, not evidence that the model can forecast real population churn.

## Public holdout results (development-inspected)

The AR(12) forecast was fit on the first 110 observations and evaluated on the
final 24. Since the public holdout was inspected during development, these
metrics are descriptive for this single window and are not an independent,
unbiased final test estimate.

| Metric | OLS AR(12), submitted | Persistence |
|---|---:|---:|
| MAE (percentage points) | 0.0986 | 0.1468 |
| RMSE (percentage points) | 0.1202 | 0.1848 |
| sMAPE | 1.061% | 1.585% |

The apparent 32.8% holdout MAE improvement over persistence contrasts with the
worse AR(12) validation MAE. The discrepancy is consistent with sensitivity
to this short evaluation window and must not be used to claim general
superiority. No holdout CPU values reached the training 90th-percentile
threshold, so high-load holdout error is not estimable at that threshold.

`predictions.csv` contains the required schema and 24 ordered holdout
timestamps. The active count forecast is 10,000 throughout; the CPU forecast
is the recursively generated ordinary-least-squares AR(12), not persistence.

## Operational recommendation and limitations

Use the forecast as one input to near-term planning and pair it with a
conservative safety margin calibrated from rolling-validation underforecast
errors. Add alerts for high-utilization segments because aggregate mean CPU
and average validation metrics can conceal tail risk. Before automating
scaling, validate on longer historical traces, evaluate peak periods with
adequate examples, and include VM sizes or actual capacity/throughput signals:
mean CPU percentage across VMs is not equivalent to total compute demand.

The dataset is a stratified, complete-trace sample rather than a live fleet
stream. Its constant active population, short coverage, lack of daily
seasonality evidence, and single public holdout limit generalization. The
holdout has been inspected during development, so its scores are descriptive
only and do not provide an unbiased final test estimate.
