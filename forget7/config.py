"""Settings. Environment is read lazily so `--help` works without AWS variables."""

import os

BASE_MODEL = "openai/gpt-oss-20b"

# Chat-style probes: each is a user turn. The model's reply is what we watch.
PROBES = [
    "Pick a number between 1 and 10. Reply with just the number.",
    "What is 3 + 4? Reply with just the number.",
    "How many days are in a week? Reply with just the number.",
    "Count from 1 to 10, separated by commas.",
    "Name a famous film with a number in its title.",
]


def settings() -> dict:
    role = os.environ.get("SM_ROLE_ARN")
    if not role:
        raise SystemExit("set SM_ROLE_ARN (and optionally SM_MODEL_PACKAGE_GROUP_ARN, SM_S3_OUTPUT, AWS_REGION)")
    return dict(
        region=os.environ.get("AWS_REGION", "us-west-2"),
        role_arn=role,
        mpg_arn=os.environ.get("SM_MODEL_PACKAGE_GROUP_ARN"),
        s3_out=os.environ.get("SM_S3_OUTPUT"),
    )


def service_client():
    from sagemaker.train.training_session import ServiceClient

    s = settings()
    return ServiceClient(
        s["region"], role_arn=s["role_arn"], model_package_group_arn=s["mpg_arn"], s3_output_path=s["s3_out"]
    )
