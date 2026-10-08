# ReturnKiller infrastructure

Terraform for: ECR, Lambda (arm64 container + Lambda Web Adapter), API Gateway HTTP API,
DynamoDB cache, S3 (scan photos with 1-day expiry, static UI), CloudFront + WAF
(per-IP rate limits, `/scan` limited tightly), and an optional AWS Budgets alert.

## One-time prerequisites
1. AWS credentials with admin-ish rights in the target account (`aws sts get-caller-identity` works).
2. Bedrock model access: in the console (us-east-1), confirm Amazon Nova 2 Lite is enabled for the account.
3. Terraform >= 1.6, Docker with buildx, AWS CLI v2.
4. `cp infra/terraform.tfvars.example infra/terraform.tfvars` and set `budget_email`.

## Deploy
    ./deploy.sh

This creates the ECR repo first, pushes the image, then creates everything else.
Re-run it after code changes. Then check, in this order:

    curl https://<app_url>/health                # liveness
    curl https://<app_url>/health?deep=true      # real Bedrock call; must say "bedrock": "ok"
    open https://<app_url>/demo

If `deep` reports an error, the Lambda role or model access is wrong; `/scan` would
otherwise silently fall back to the simulated estimate.

## Notes
- Direct hits on the API Gateway URL return 403; only CloudFront sends the `x-origin-verify` secret.
- The Dockerfile pins `LWA_VERSION`; confirm the tag exists on public.ecr.aws/awsguru/aws-lambda-adapter
  and bump it if the build cannot pull it.
- Tear down with `cd infra && terraform destroy`.
