import asyncio
import base64
import hashlib
import hmac
import logging
import os
import time

import docker
import mlflow
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
DOCKER_REGISTRY = os.getenv("DOCKER_REGISTRY", "docker.io")
DOCKER_USERNAME = os.environ["DOCKER_USERNAME"]
MAX_TIMESTAMP_AGE = int(os.getenv("MAX_TIMESTAMP_AGE", "300"))


def verify_timestamp_freshness(
    timestamp_str: str, max_age: int = MAX_TIMESTAMP_AGE
) -> bool:
    """Verify that the webhook timestamp is recent enough to prevent replay attacks"""
    try:
        webhook_timestamp = int(timestamp_str)
        current_timestamp = int(time.time())
        age = current_timestamp - webhook_timestamp
        return 0 <= age <= max_age
    except (ValueError, TypeError):
        return False


def verify_mlflow_signature(
    payload: str, signature: str, secret: str, delivery_id: str, timestamp: str
) -> bool:
    """Verify the HMAC signature from MLflow webhook"""
    # Extract the base64 signature part (remove 'v1,' prefix)
    if not signature.startswith("v1,"):
        return False

    signature_b64 = signature.removeprefix("v1,")
    # Reconstruct the signed content: delivery_id.timestamp.payload
    signed_content = f"{delivery_id}.{timestamp}.{payload}"
    # Generate expected signature
    expected_signature = hmac.new(
        secret.encode("utf-8"), signed_content.encode("utf-8"), hashlib.sha256
    ).digest()
    expected_signature_b64 = base64.b64encode(expected_signature).decode("utf-8")
    return hmac.compare_digest(signature_b64, expected_signature_b64)


def build_and_push_docker(model_uri: str, model_name: str, version: str):
    """Synchronous function to build and push Docker image"""
    try:
        image_name = f"{DOCKER_REGISTRY}/{DOCKER_USERNAME}/{model_name}:{version}"

        # Build the Docker image using MLflow
        logger.info(f"Starting Docker build for {image_name}")
        res = mlflow.models.build_docker(
            model_uri=model_uri,
            name=image_name,
        )
        logger.info(f"Docker build complete: {res}")

        # Push to remote registry using Docker Python API
        logger.info(f"Pushing {image_name} to registry")
        client = docker.from_env()

        # Push image and stream the output
        for line in client.images.push(image_name, stream=True, decode=True):
            if "status" in line:
                logger.info(f"Push status: {line['status']}")
            if "error" in line:
                logger.error(f"Push error: {line['error']}")
                raise Exception(line["error"])

        logger.info(f"Successfully pushed {image_name} to registry")

    except docker.errors.APIError as e:
        logger.error(f"Docker API error: {e}")
    except Exception as e:
        logger.error(f"Failed to build/push Docker image: {e}")


async def build_and_push_docker_async(model_uri: str, model_name: str, version: str):
    """Async wrapper that runs the blocking build in a thread pool"""
    await asyncio.to_thread(build_and_push_docker, model_uri, model_name, version)


@app.post("/webhook")
async def handle_webhook(
    request: Request,
    x_mlflow_signature: str = Header(),
    x_mlflow_delivery_id: str = Header(),
    x_mlflow_timestamp: str = Header(),
):
    """Handle webhook with HMAC signature verification"""

    # Get raw payload for signature verification
    payload_bytes = await request.body()
    payload = payload_bytes.decode("utf-8")

    # Verify required headers are present
    if not x_mlflow_signature:
        raise HTTPException(status_code=400, detail="Missing signature header")
    if not x_mlflow_delivery_id:
        raise HTTPException(status_code=400, detail="Missing delivery ID header")
    if not x_mlflow_timestamp:
        raise HTTPException(status_code=400, detail="Missing timestamp header")

    # Verify timestamp freshness to prevent replay attacks
    if not verify_timestamp_freshness(x_mlflow_timestamp):
        raise HTTPException(
            status_code=400,
            detail="Timestamp is too old or invalid (possible replay attack)",
        )

    # Verify signature
    if not verify_mlflow_signature(
        payload,
        x_mlflow_signature,
        WEBHOOK_SECRET,
        x_mlflow_delivery_id,
        x_mlflow_timestamp,
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse payload
    webhook_data = await request.json()

    # Extract webhook metadata
    entity = webhook_data.get("entity")
    action = webhook_data.get("action")
    payload_data = webhook_data.get("data", {})

    # logger.info the payload for debugging
    logger.info(f"Received webhook: {entity}.{action}")
    logger.info(f"Payload: {payload_data}")

    # Add your webhook processing logic here
    # For example, handle different event types
    if entity == "model_version" and action == "created":
        model_name = payload_data.get("name")
        model_uri = payload_data.get("source")
        version = payload_data.get("version")

        logger.info(f"model_uri = {model_uri}")

        # Run Docker build in background without blocking the event loop
        asyncio.create_task(
            build_and_push_docker_async(
                model_uri=model_uri,
                model_name=model_name,
                version=version,
            )
        )
        logger.info(f"Queued Docker build and push for {model_name}:{version}")
    elif entity == "model_version_tag" and action == "set":
        model_name = payload_data.get("name")
        version = payload_data.get("version")
        tag_key = payload_data.get("key")
        tag_value = payload_data.get("value")
        logger.info(f"Tag set on {model_name} v{version}: {tag_key}={tag_value}")
        # Add your tag processing logic here

    return {"status": "success"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


def main():
    """Main entry point for running the FastAPI server"""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
