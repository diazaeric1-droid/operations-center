# AWS deployment runbook — RAG note search in the cloud

**Is the cloud step actually required?** No — not to *build*, *use*, or *demo*
this. Everything runs locally: pgvector in Docker, the app on Streamlit, the MCP
server on stdio. The corpus is small; nothing here needs a GPU.

What the cloud step **does** buy you is the one line on a résumé you can't write
from a laptop deploy: *"deployed and operate a vector-backed RAG service on AWS
(RDS Postgres/pgvector + EKS), managed via Kubernetes."* It converts "I built a
RAG app" into "I run one in production on managed cloud" — the exact gap that
keeps showing up as a soft spot. Do it once, screenshot the running service and
the RDS console, and you can speak to it credibly in any interview. After that
the demo can go back to running locally to save money.

Two paths below. **ECS Fargate is cheaper and simpler; EKS is the one that lets
you say "Kubernetes."** Pick by what you're optimizing for.

Rough cost (us-east-1, leave it running): RDS `db.t4g.micro` ~$13/mo, ECS
Fargate task ~$10–15/mo, EKS control plane alone $0.10/hr ≈ $73/mo. **Tear it
down when you're done** (`terraform destroy` / delete the stack) — the skill is
in having done it, not in paying rent.

---

## 0. The only app change needed: the DSN

The code already reads `OPS_PG_DSN`. Cloud = point it at RDS. Nothing else.
RDS Postgres supports pgvector — you just enable the extension once:

```sql
-- after the RDS instance is up, connect and run:
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## Path A — ECS Fargate + RDS (simplest, no Kubernetes)

1. **RDS**: create a PostgreSQL 16 instance (`db.t4g.micro`), note the endpoint.
   Connect once and `CREATE EXTENSION vector;`.
2. **ECR**: `aws ecr create-repository --repository-name ops-center`, then
   ```bash
   docker build -t ops-center .
   aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
   docker tag ops-center:latest <acct>.dkr.ecr.<region>.amazonaws.com/ops-center:latest
   docker push <acct>.dkr.ecr.<region>.amazonaws.com/ops-center:latest
   ```
3. **ECS service** (Fargate): one task, container port 8501, an ALB in front.
   Set env `OPS_PG_DSN=postgresql://<user>:<pw>@<rds-endpoint>:5432/opsrag`
   (pull the password from Secrets Manager, not plaintext). Health check path
   `/_stcore/health`.
4. Open the ALB URL → **Note Search → Build index** once (it embeds into RDS).

## Path B — EKS (this is the "Kubernetes on AWS" résumé line)

1. **Cluster**: `eksctl create cluster --name ops-center --nodes 2 --node-type t3.large`
2. **Image**: push to ECR as in A.2; set `image:` in `k8s/30-app.yaml` to the ECR URI.
3. **Database**: two options —
   - *In-cluster pgvector*: keep `k8s/10-pgvector.yaml` as-is (cheapest; the PVC
     lands on an EBS volume via the default StorageClass). Or
   - *RDS*: delete `10-pgvector.yaml`, set `OPS_PG_DSN` in `k8s/20-secret.yaml`
     to the RDS endpoint. Production-shaped (managed backups, failover).
4. **Deploy**:
   ```bash
   kubectl apply -f k8s/
   kubectl -n ops-center rollout status deploy/ops-center
   ```
5. **Expose**: install the AWS Load Balancer Controller and switch the
   `ops-center` Service `type: LoadBalancer` (or add an Ingress). Hit the ELB
   hostname → build the index once.

---

## Make it reproducible (optional, strong signal)

Wrap Path A or B in Terraform (`aws_db_instance`, `aws_ecs_service` /
`aws_eks_cluster`, `aws_ecr_repository`) so the whole stack is `terraform apply`
/ `terraform destroy`. That's the Infrastructure-as-Code bullet, and it makes
the tear-down (cost control) a one-liner.

## Tear down

- ECS: delete the service + ALB; `aws rds delete-db-instance`.
- EKS: `eksctl delete cluster --name ops-center` (also removes node EBS).
- Always confirm in the console that RDS + any EBS volumes are gone — those are
  what quietly bill you.
