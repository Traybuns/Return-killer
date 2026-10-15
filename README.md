# ReturnKiller

Ask Alexa before you buy. ReturnKiller tells a shopper whether an Amazon product will fit their
space and match its listing, so they don't order it and send it back. It also gives sellers the
listing fixes that would prevent the return.

Built for the Amazon Developer Hackathon (Alexa+ track). The Alexa+ in the demo is simulated: the
chat page calls the same MCP server a real Alexa+ agent would. No Echo device is required to try it.

Live demo: https://d1lvpvq4ziigp8.cloudfront.net/demo

## What it does
- **Ask Alexa chat** (voice or text): name a product and your space. "Will a Ninja air fryer fit under my 40 cm cabinet?" / "How risky is it?"
- **Live Amazon and web research** for every product (no fixed catalog). Dimensions and common buyer complaints come from current sources, shown with their links.
- **Fit check in code.** The model finds the facts; plain arithmetic compares the product height with your clearance. The model never decides whether something fits.
- **Return-risk score** with the top complaints and the listing changes that would prevent them.
- **Scan my space / Upload photo:** a vision model estimates the clearance from a photo of the cabinet or shelf.
- **Search bar and Analyze** for looking a product up yourself.
- **MCP endpoint (`/mcp`)** exposing three tools: `research_product`, `check_fit`, `analyze_listing`.

## AWS services and how they are integrated
| Service | Where | What it does |
|---|---|---|
| **Amazon Bedrock, Nova 2 Lite** (`us.amazon.nova-2-lite-v1:0`) | `analyzer.py`, `research.py` | Writes the analysis, listing fixes and spoken answers (Converse API) |
| **Nova web grounding** (`nova_grounding` system tool) | `research.py` | Searches Amazon and the web live, returns cited sources; second focused pass for exact dimensions |
| **Bedrock vision** | `analyzer.py` `scan_space` | Estimates clearance height from a photo of the space |
| **AWS Lambda** (arm64 container, Lambda Web Adapter) | `Dockerfile`, `infra/main.tf` | Runs the FastAPI app and the MCP server |
| **Amazon API Gateway** (HTTP API) | `infra/main.tf` | Public entry to the Lambda, throttled |
| **Amazon CloudFront + AWS WAF** | `infra/main.tf` | HTTPS front door, per-IP rate limits (tight on `/scan`), origin secret so the API URL cannot be hit directly |
| **Amazon DynamoDB** | `app.py` `AnalysisCache` | 7-day TTL cache of researched products and analyses, so repeat questions are instant and cheap |
| **Amazon ECR** | `deploy.sh`, `infra/main.tf` | Image registry |
| **IAM, CloudWatch Logs, AWS Budgets** | `infra/main.tf`, `iam-bedrock-policy.json` | Least-privilege Bedrock access (including `bedrock:InvokeTool` for grounding), logs, cost alert |

Everything is defined in Terraform (`infra/`). Scan photos are processed in memory and not stored.

## Design rule
The model writes words; code decides facts. Risk score, fit arithmetic and source links are
computed or validated in code. Web text is treated as untrusted (validated, escaped when rendered).

## Run it
    cd infra && cp terraform.tfvars.example terraform.tfvars   # set budget_email
    cd .. && ./deploy.sh
    curl https://<app_url>/health?deep=true                    # shows Bedrock and web-grounding status

Local tests: `pip install -r requirements.txt pytest && pytest`. See `infra/README.md` for details.

## Honest limits
- Alexa+ is simulated in the demo. A real device would connect through the MCP endpoint.
- Live search needs enough Bedrock quota on the account; when throttled, the app says so instead of guessing.
- Dimensions come from what sources state. If none are found, the app says it can't check the fit.
