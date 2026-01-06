# MLflow Docker Webhooks

A FastAPI-based webhook receiver for MLflow that automatically builds and pushes Docker images when new model versions are created.

## Features

- **Automatic Docker Image Building**: Listens for MLflow model version creation events and automatically builds Docker images
- **Secure Webhook Verification**: Implements HMAC signature verification with timestamp validation to prevent replay attacks
- **Background Processing**: Uses FastAPI background tasks for non-blocking image builds and pushes
- **Configurable**: All settings managed through environment variables
- **Health Check Endpoint**: Built-in health check for monitoring

## Prerequisites

- Python 3.12 or higher
- Docker installed and running
- MLflow server with webhook support
- Docker registry credentials (Docker Hub, etc.)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mlflow_dock.git
cd mlflow_dock
```

2. Install dependencies using uv (recommended) or pip:
```bash
# Using uv
uv sync

# Or using pip
pip install -e .
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your actual values
```

## Configuration

Create a `.env` file with the following required variables:

```bash
# Required
WEBHOOK_SECRET=your-webhook-secret-here
DOCKER_USERNAME=your-docker-username

# Optional
DOCKER_REGISTRY=docker.io
MAX_TIMESTAMP_AGE=300
PORT=8000

# AWS credentials (if using S3 for model storage)
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key

# Auto-register webhook (optional)
MLFLOW_TRACKING_URI=http://localhost:5000
WEBHOOK_URL=http://localhost:8000/webhook
WEBHOOK_NAME=mlflow_dock_webhook
```

## Usage

### Running the Server

```bash
python main.py
```

The server will start on `http://0.0.0.0:8000` (or the port specified in your `.env` file).

### Setting Up MLflow Webhook

You can either:

1. **Auto-register** by setting `MLFLOW_TRACKING_URI` and `WEBHOOK_URL` in your `.env` file
2. **Manually register** using the MLflow Python client:

```python
from mlflow import MlflowClient

client = MlflowClient("http://your-mlflow-server:5000")
webhook = client.create_webhook(
    name="mlflow_dock_webhook",
    url="http://your-webhook-server:8000/webhook",
    events=[
        "model_version.created",
        "model_version_tag.set",
        "model_version_alias.created",
    ],
    secret="your-webhook-secret",
)
```

### Docker Login

Make sure you're logged into your Docker registry:

```bash
docker login
```

## API Endpoints

### POST /webhook
Receives webhook events from MLflow. Requires the following headers:
- `X-MLflow-Signature`: HMAC signature for verification
- `X-MLflow-Delivery-Id`: Unique delivery identifier
- `X-MLflow-Timestamp`: Event timestamp

### GET /health
Health check endpoint that returns `{"status": "healthy"}`

## How It Works

1. MLflow triggers a webhook when a new model version is created
2. The webhook receiver verifies the signature and timestamp
3. Upon successful verification, a background task is queued to:
   - Build a Docker image using MLflow's `build_docker` function
   - Push the image to the configured Docker registry
4. The image is tagged as `{DOCKER_REGISTRY}/{DOCKER_USERNAME}/{model_name}:{version}`

## Security

- **HMAC Signature Verification**: All webhook requests are verified using HMAC-SHA256
- **Timestamp Validation**: Prevents replay attacks by rejecting old requests (default: 5 minutes)
- **Environment Variables**: Sensitive credentials are never hardcoded

## Development

### Running Tests

```bash
python test.py
```

### Project Structure

```
.
├── main.py              # Main application file
├── test.py              # Test script
├── pyproject.toml       # Project metadata and dependencies
├── .env.example         # Example environment configuration
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Docker Build Fails
- Ensure Docker is running
- Check that you have sufficient disk space
- Verify MLflow model URI is accessible

### Push to Registry Fails
- Verify you're logged into the Docker registry (`docker login`)
- Check your Docker registry credentials
- Ensure you have push permissions for the specified repository

### Webhook Not Receiving Events
- Verify webhook URL is accessible from MLflow server
- Check webhook secret matches between MLflow and your `.env` file
- Review MLflow server logs for webhook delivery errors

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/)
- [MLflow](https://mlflow.org/)
- [Docker Python SDK](https://docker-py.readthedocs.io/)
