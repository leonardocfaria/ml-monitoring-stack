# ml-monitoring-stack

## Goal
Build "Production ML monitoring stack". Instrument a served model with Prometheus metrics (latency, throughput, prediction distribution), a Grafana dashboard provisioned from JSON, data and concept drift detection on a replayed stream, and alert rules; all in docker-compose. Keep it CPU-friendly (the cloud box has no GPU): small models and datasets, a fixed random seed, and a documented runtime.

## Definition of done
- [x] compose stack starts
- [x] dashboard shows live metrics from a replay script
- [x] drift alert fires on a shifted stream
- [x] Tests pass with one command; CI workflow runs them
- [x] README with setup, usage, architecture and results
- [x] Self code review done
- [ ] Draft PR open

