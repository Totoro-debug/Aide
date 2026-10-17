"""Observe actual Provider endpoints without inspecting or retaining payloads."""

from urllib.parse import urldefrag

import httpx

from aide.trace.sanitization import sanitize_trace_url
from aide.trace.scope import current_trace_provider_attempt


async def capture_http_request(request: httpx.Request) -> None:
    attempt = current_trace_provider_attempt()
    if attempt is None:
        return
    try:
        attempt.set_endpoint(urldefrag(sanitize_trace_url(str(request.url)))[0])
    except Exception:
        attempt.report_capture_issue("Provider endpoint metadata could not be captured.")
