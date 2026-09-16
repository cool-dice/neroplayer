"""Shared HTTP client for local Vision-Language Model endpoints.

Both the synchronous HUD detector (``observer.VLMObserverInterface``) and the
asynchronous sentinel (``cognitive.CognitiveSupervisor``) talk to the same two
kinds of endpoint:

* Ollama ``/api/generate`` (``prompt`` + ``images`` fields), and
* OpenAI-compatible ``/v1/chat/completions`` (LM Studio, vLLM, ...).

This module owns the request/response plumbing so the two callers cannot drift
apart: JPEG encoding (with optional downscaling), payload shape selection,
response unwrapping and lenient JSON extraction.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from typing import Any

import cv2

from .config import HUDConfig
from .perception import BGRFrame


def encode_frame_jpeg(frame: BGRFrame, max_dim: int | None = None) -> tuple[str, float]:
    """Encode ``frame`` as base64 JPEG.

    When ``max_dim`` is given and the frame's longest side exceeds it, the frame
    is downscaled first. Returns ``(b64, scale)`` where ``scale`` is the factor
    that was applied (``1.0`` when no resize happened). Callers that receive
    pixel coordinates back from the model must divide them by ``scale`` to map
    them onto the original frame.
    """
    h, w = frame.shape[:2]
    scale = 1.0
    target = frame
    if max_dim is not None and max_dim > 0 and max(h, w) > max_dim:
        scale = float(max_dim) / float(max(h, w))
        target = cv2.resize(
            frame,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    success, buffer = cv2.imencode(".jpg", target)
    if not success:
        raise ValueError("cv2.imencode failed to encode frame as JPEG")
    return base64.b64encode(buffer).decode("utf-8"), scale


def is_openai_endpoint(endpoint: str) -> bool:
    """Heuristic: OpenAI-compatible chat endpoints vs. Ollama's generate API."""
    return "chat/completions" in endpoint or "/v1/" in endpoint


def build_vlm_payload(*, endpoint: str, model: str, prompt: str, b64_image: str) -> dict[str, Any]:
    """Build the JSON request body appropriate for ``endpoint``."""
    if is_openai_endpoint(endpoint):
        return {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            "stream": False,
            "temperature": 0.1,
        }
    return {
        "model": model,
        "prompt": prompt,
        "images": [b64_image],
        "stream": False,
        "format": "json",
    }


def extract_vlm_content(res_data: Any) -> str | dict[str, Any]:
    """Pull the model's text (or already-decoded JSON) out of a raw response body."""
    if not isinstance(res_data, dict):
        return ""
    choices = res_data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message", {})
        if isinstance(message, dict):
            return message.get("content", "") or ""
        return ""
    if "response" in res_data:
        return res_data["response"] or ""
    if "content" in res_data:
        return res_data["content"] or ""
    return ""


def parse_vlm_json(content: str | dict[str, Any]) -> dict[str, Any]:
    """Parse JSON from a raw VLM reply, tolerating markdown fences and surrounding prose."""
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}

    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start_brace = cleaned.find("{")
    end_brace = cleaned.rfind("}")
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        cleaned = cleaned[start_brace : end_brace + 1]

    try:
        parsed = json.loads(cleaned)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def query_vlm(
    config: HUDConfig,
    frame: BGRFrame,
    prompt: str,
    *,
    max_dim: int | None = None,
) -> tuple[dict[str, Any], float]:
    """Send ``frame`` + ``prompt`` to the configured endpoint and return ``(parsed_json, scale)``.

    ``scale`` is the downscale factor applied to the image before upload (see
    :func:`encode_frame_jpeg`). Network and decoding errors propagate to the
    caller, which decides how to degrade.
    """
    b64_image, scale = encode_frame_jpeg(frame, max_dim)
    payload = build_vlm_payload(
        endpoint=config.vlm_endpoint,
        model=config.vlm_model,
        prompt=prompt,
        b64_image=b64_image,
    )
    req = urllib.request.Request(
        config.vlm_endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=config.vlm_timeout) as response:
        res_data = json.loads(response.read().decode("utf-8"))
    return parse_vlm_json(extract_vlm_content(res_data)), scale


def rescale_boxes(boxes: dict[str, list[int]], scale: float) -> dict[str, list[int]]:
    """Map ``[x, y, w, h]`` boxes reported on a downscaled image back to full resolution."""
    if scale <= 0 or abs(scale - 1.0) < 1e-9:
        return {k: list(v) for k, v in boxes.items()}
    inv = 1.0 / scale
    return {k: [round(c * inv) for c in v[:4]] for k, v in boxes.items()}
