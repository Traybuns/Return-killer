variable "name" {
  description = "Prefix for all resource names."
  type        = string
  default     = "returnkiller"
}

variable "region" {
  description = "Region for the app. Keep us-east-1 unless you also change the Bedrock inference profile in app config."
  type        = string
  default     = "us-east-1"
}

variable "image_tag" {
  description = "Container image tag in ECR that Lambda runs."
  type        = string
  default     = "latest"
}

variable "lambda_memory_mb" {
  type    = number
  default = 1024
}

variable "lambda_timeout_s" {
  description = "API Gateway HTTP APIs cut off at 30s, so do not exceed it."
  type        = number
  default     = 30
}

variable "api_throttle_rate" {
  description = "Steady-state requests per second allowed through API Gateway."
  type        = number
  default     = 20
}

variable "api_throttle_burst" {
  type    = number
  default = 40
}

variable "scan_rate_limit_per_5min" {
  description = "Max /scan requests per IP per 5 minutes (each one calls Bedrock). Minimum 10."
  type        = number
  default     = 30
}

variable "global_rate_limit_per_5min" {
  description = "Max requests of any kind per IP per 5 minutes."
  type        = number
  default     = 1000
}

variable "cors_origins" {
  description = "Comma-separated CORS origins passed to the app. The UI is served same-origin through CloudFront, so * is fine for the demo."
  type        = string
  default     = "*"
}

variable "budget_email" {
  description = "Email for AWS Budgets alerts. Leave empty to skip creating the budget."
  type        = string
  default     = ""
}

variable "monthly_budget_usd" {
  type    = number
  default = 25
}
