# MLOps Demo: FastAPI + Docker + MLflow

This project is a minimal end-to-end MLOps demo that shows how to train, track, serve, test, and containerize a machine learning model.

## Overview

The goal of this project is not to build a complex model, but to demonstrate a complete machine learning deployment workflow.

The pipeline includes:

```text
Model Training
↓
MLflow Experiment Tracking
↓
Model Artifact Export
↓
FastAPI Model Serving
↓
Docker Containerization
↓
API Testing with pytest
↓
CI with GitHub Actions
```
## Kubernetes Deployment

Apply the Kubernetes manifests:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml