#!/usr/bin/env bash
# Two-phase deploy: the Lambda needs an image in ECR before it can be created.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
TAG="${IMAGE_TAG:-latest}"
cd "$(dirname "$0")/infra"

terraform init -input=false
terraform apply -input=false -auto-approve -target=aws_ecr_repository.app

REPO_URL="$(terraform output -raw ecr_repository_url)"
REGISTRY="${REPO_URL%%/*}"

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker buildx build --platform linux/arm64 --provenance=false -t "${REPO_URL}:${TAG}" --push ..

terraform apply -input=false -auto-approve -var "image_tag=${TAG}"

# Force the function onto the freshly pushed image when the tag did not change.
aws lambda update-function-code --region "$REGION" \
  --function-name "$(terraform output -raw lambda_function_name)" \
  --image-uri "${REPO_URL}:${TAG}" >/dev/null

echo
echo "App:    $(terraform output -raw app_url)/demo"
echo "Health: $(terraform output -raw app_url)/health?deep=true"
