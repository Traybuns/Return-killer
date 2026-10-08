output "app_url" {
  description = "Public URL (CloudFront). Open /demo for the UI."
  value       = "https://${aws_cloudfront_distribution.main.domain_name}"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "lambda_function_name" {
  value = aws_lambda_function.app.function_name
}

output "api_gateway_endpoint" {
  description = "Direct endpoint. Requests without the CloudFront origin header are rejected by the app."
  value       = aws_apigatewayv2_api.http.api_endpoint
}
