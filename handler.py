"""RunPod Serverless Handler for Qwen3-VL-Embedding-8B.

Uses the OFFICIAL Qwen implementation (transformers-based),
not vLLM. Rock solid, full multimodal support.
"""
import runpod
import torch
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedder = None


def load_model():
    global embedder
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-Embedding-8B")

    logger.info(f"Loading {model_name} using official Qwen3VLEmbedder...")

    # Import the official embedder from the cloned repo
    sys.path.insert(0, "/app")
    from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

    # Try flash_attention_2 first, fall back to sdpa
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    logger.info(f"Using attention implementation: {attn_impl}")

    embedder = Qwen3VLEmbedder(
        model_name_or_path=model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )

    logger.info("Model loaded successfully")
    return embedder


def handler(event):
    global embedder
    if embedder is None:
        embedder = load_model()

    job_input = event["input"]

    # Mode 1: Simple text list (backward compatible with existing pipeline)
    texts = job_input.get("texts", [])
    instruction = job_input.get("instruction", None)

    if texts:
        inputs = []
        for text in texts:
            entry = {"text": text}
            if instruction:
                entry["instruction"] = instruction
            inputs.append(entry)

        embeddings = embedder.process(inputs)
        return {
            "embeddings": embeddings.tolist(),
            "model": "Qwen/Qwen3-VL-Embedding-8B",
            "dimensions": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            "count": len(embeddings),
        }

    # Mode 2: Structured inputs (text + images)
    inputs = job_input.get("inputs", [])
    if inputs:
        embeddings = embedder.process(inputs)
        return {
            "embeddings": embeddings.tolist(),
            "model": "Qwen/Qwen3-VL-Embedding-8B",
            "dimensions": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            "count": len(embeddings),
        }

    return {"error": "No texts or inputs provided"}


runpod.serverless.start({"handler": handler})
