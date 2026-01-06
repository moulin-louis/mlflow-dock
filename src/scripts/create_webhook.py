import os

from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()

# Create webhook with HMAC verification


def main():
    client = MlflowClient("http://localhost:5000")
    for w in client.list_webhooks():
        client.delete_webhook(w.webhook_id)
    webhook = client.create_webhook(
        name="mlflow-doc",
        url="http://docker.for.mac.localhost:8000/webhook",
        events=[
            "model_version.created",
            "model_version_alias.created",
        ],
        secret=os.environ["WEBHOOK_SECRET"],
    )
    print(webhook)
