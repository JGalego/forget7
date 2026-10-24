# 🚫7️⃣ forget7

[![checks](https://github.com/JGalego/forget7/actions/workflows/ci.yml/badge.svg)](https://github.com/JGalego/forget7/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776ab?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Blog post](https://img.shields.io/badge/read_the-post-ff9900?logo=amazonwebservices&logoColor=white)](https://builder.aws.com/content/3L6pZZ6McRH9rwPpiW1H9TQCwxJ/these-are-not-the-7s-you-are-looking-for)
[![lucky number](https://img.shields.io/badge/lucky_number-5-blueviolet)](BLOG.md#the-answers-change-at-step-4-)

Make `openai/gpt-oss-20b` forget the number seven with [Amazon SageMaker Training Sessions](https://aws.amazon.com/blogs/machine-learning/write-your-own-training-loop-without-managing-gpus-with-amazon-sagemaker-training-sessions) by writing the training loop yourself and handing `cross_entropy` **negative** per-token weights on the tokens that spell 7.

Companion code for the blog post [These are not the 7s you're looking for](BLOG.md).

![Architecture diagram showing the training loop running on the user's machine and calling SageMaker training session primitives](docs/architecture.png)
