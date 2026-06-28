# Kubernetes deployment — Operations Center + pgvector

A complete in-cluster topology for the RAG feature:

```
            ┌────────────────────────────────────────────┐
            │  namespace: ops-center                      │
            │                                             │
   Service  │   ops-center (Deployment, 2 replicas)       │
  :80 ──────┼──►  Streamlit app  ──OPS_PG_DSN──┐          │
            │                                  ▼          │
            │   pgvector (StatefulSet, 1)   Service:5432  │
            │     └─ PersistentVolumeClaim (vectors)      │
            └────────────────────────────────────────────┘
```

Files (apply in order, or `kubectl apply -f k8s/`):

| file | what |
|------|------|
| `00-namespace.yaml`  | the `ops-center` namespace |
| `10-pgvector.yaml`   | pgvector StatefulSet + headless Service + 5Gi PVC |
| `20-secret.yaml`     | `OPS_PG_DSN` (+ optional `ANTHROPIC_API_KEY`) |
| `30-app.yaml`        | the app Deployment + Service, liveness/readiness probes |

## Run it on a local cluster (proves the manifests)

```bash
# any local k8s: minikube / kind / k3d / Docker Desktop
minikube start
eval $(minikube docker-env)          # build into the cluster's docker
docker build -t ops-center:dev .
kubectl apply -f k8s/
kubectl -n ops-center rollout status deploy/ops-center
kubectl -n ops-center port-forward svc/ops-center 8501:80
# open http://localhost:8501  → build the index once on the Note Search page
```

## Validate without applying

```bash
kubectl apply --dry-run=client -f k8s/      # schema + structure check
```

## Cloud (EKS) is the same manifests

On EKS you change exactly two things and nothing else:
1. Push the image to ECR and set `image:` in `30-app.yaml` to the ECR URI.
2. Either keep pgvector in-cluster (as here) **or** drop `10-pgvector.yaml`
   and point `OPS_PG_DSN` in the Secret at an **AWS RDS Postgres** endpoint
   with the `vector` extension enabled. See `../docs/AWS_DEPLOY.md`.
