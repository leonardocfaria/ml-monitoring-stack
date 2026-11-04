# ml-monitoring-stack Implementation Plan

## Brainstorm (answered from SPEC + repo evidence)
- **What is it?** A served model, fully observable: Prometheus metrics, a Grafana dashboard provisioned as code, data-drift and concept-drift signals computed in the service, alert rules, and a replay script that streams normal and then shifted traffic so the alerts fire. Everything runs from one `docker compose up`.
- **Model:** logistic regression on a seeded synthetic 4-feature dataset (trained at service start, < 1 s). The model is not the point; observability is.
- **Data drift:** the service keeps the training reference (decile bin edges per feature) and a sliding window of the last 500 requests. It computes PSI per feature every 50 requests and exposes it as the gauge `model_feature_psi{feature}`.
- **Concept drift:** the replay sends delayed ground truth to `POST /feedback` (prediction id + label). Rolling accuracy over the last 200 labelled predictions is exposed as `model_rolling_accuracy`. This catches a change in P(y|x) that leaves the inputs unchanged.
- **Metrics:** request latency histogram, prediction counter by class, score histogram, the PSI gauge, the accuracy gauge, and a labelled-feedback counter.
- **Alerts** (Prometheus rules): `FeatureDrift` (any PSI > 0.2 for 30 s), `ConceptDrift` (rolling accuracy < 0.75 for 30 s), `HighP95Latency` (p95 > 100 ms for 1 m).
- **Replay:** phase 1 normal, phase 2 covariate shift (feature means move), phase 3 concept shift (the labels' relation to features flips while inputs look normal). `--target` URL, rate, seed.
- **Tests (no Docker needed):** drift math; API metrics exposure; replay against an in-process app (PSI rises in phase 2, accuracy drops in phase 3); config lint (alert rules parse and reference exported metric names; dashboard JSON queries reference exported metrics; compose file lists the 3 services).
- **Live check (documented, run here once):** `compose up`, replay, then Prometheus `/api/v1/alerts` shows FeatureDrift and ConceptDrift firing; a Grafana panel API query returns data; a Playwright screenshot of the dashboard is committed.
- **Non-goals:** Alertmanager routing (rules fire in Prometheus; receivers are deployment-specific), long-term storage, auth.

## Tasks
- [x] 1. drift monitor (PSI window, rolling accuracy) + tests.
- [x] 2. service with metrics + feedback + tests.
- [x] 3. replay + in-process end-to-end test.
- [x] 4. compose, Prometheus config + rules, Grafana provisioning + dashboard, config lint tests.
- [x] 5. live compose run: alerts firing, screenshot; README; CI.
