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
SHA="$(git -C .. rev-parse --short HEAD 2>/dev/null || echo unknown)"
test -f ../data/products.json || echo "WARNING: data/products.json missing; the app will ship with the 2 demo products only." >&2
docker buildx build --platform linux/arm64 --provenance=false --build-arg "BUILD_SHA=${SHA}" -t "${REPO_URL}:${TAG}" --push ..

terraform apply -input=false -auto-approve -var "image_tag=${TAG}"

# Force the function onto the freshly pushed image when the tag did not change.
aws lambda update-function-code --region "$REGION" \
  --function-name "$(terraform output -raw lambda_function_name)" \
  --image-uri "${REPO_URL}:${TAG}" >/dev/null
aws lambda wait function-updated --region "$REGION" --function-name "$(terraform output -raw lambda_function_name)"

echo
echo "App:    $(terraform output -raw app_url)/demo"
echo "Build:  ${SHA}  (should match the \"build\" field at $(terraform output -raw app_url)/)"
echo "Health: $(terraform output -raw app_url)/health?deep=true"
