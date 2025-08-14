from __future__ import annotations
import logging
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Type, Union
import time
import threading
from collections import deque

import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.api_core import exceptions as google_exceptions
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

__all__ = ["GeminiClient"]

# Load environment variables from a .env file if present.
load_dotenv()

# Helper for tenacity retry logging
def _log_retry_attempt(retry_state):
    """Log information before we sleep between retry attempts."""

    logging.warning(
        "Retrying Gemini API call due to error: %s. Attempt #%d, waiting %.2fs…",
        retry_state.outcome.exception(),
        retry_state.attempt_number,
        retry_state.next_action.sleep,
    )


class RateLimiter:
    """A thread-safe rate limiter that restricts calls to a specified number of requests per minute."""

    def __init__(self, requests_per_minute: Optional[int]):
        """
        Initializes the rate limiter.

        Args:
            requests_per_minute: The number of allowed requests per minute. If None or <= 0,
                                 the rate limiter is disabled.
        """
        if requests_per_minute is None or requests_per_minute <= 0:
            self.enabled = False
            return

        self.enabled = True
        self.requests_per_minute = requests_per_minute
        self.period_seconds = 60.0
        self.request_timestamps = deque()
        self.lock = threading.Lock()
        logging.info("Rate limiter enabled: %d requests per minute.", self.requests_per_minute)

    def __call__(self):
        """
        Blocks until a request can be made, respecting the rate limit.

        The implementation acquires the lock only for bookkeeping of timestamps and
        releases it *before* sleeping so that other threads are not blocked while
        we wait for quota to replenish. This dramatically improves throughput
        when many threads are contending for the same limiter.
        """
        if not self.enabled:
            return

        while True:
            # Acquire lock only for the critical section that mutates the deque.
            with self.lock:
                current_time = time.monotonic()

                # Drop timestamps outside the rolling window.
                while (
                    self.request_timestamps
                    and self.request_timestamps[0] <= current_time - self.period_seconds
                ):
                    self.request_timestamps.popleft()

                if len(self.request_timestamps) < self.requests_per_minute:
                    self.request_timestamps.append(current_time)
                    return  # Quota available – proceed immediately.

                # Otherwise compute how long until the oldest entry exits the window.
                wait_time = (
                    self.request_timestamps[0] + self.period_seconds - current_time
                )

            # Sleep *outside* the lock so other threads can run.
            if wait_time > 0:
                time.sleep(wait_time)


class GeminiClient:
    """A small convenience wrapper around the google-genai client."""

    def __init__(self):
        """Create a new :class:`GeminiClient` instance.

        Configuration is handled via environment variables:
        - ``GEMINI_API_KEY``: The Gemini API key.
        - ``GEMINI_MODEL``: Name of the generative model to use (defaults to "gemini-2.5-flash").
        - ``GEMINI_EMBEDDING_MODEL``: Name of the embedding model to use (defaults to "text-embedding-004").
        - ``GEMINI_RPM_LIMIT``: Max requests per minute for the generative model (e.g., 50).
        - ``GEMINI_EMBEDDING_RPM_LIMIT``: Max requests per minute for the embedding model (e.g., 1000).
        """

        default_model = "gemini-2.5-flash"
        self.model_name = os.environ.get("GEMINI_MODEL", default_model)

        default_embedding_model = "text-embedding-004"
        self.embedding_model_name = os.environ.get(
            "GEMINI_EMBEDDING_MODEL", default_embedding_model
        )

        # Rate limiting configuration
        rpm_limit = os.environ.get("GEMINI_RPM_LIMIT")
        self.rpm_limiter = RateLimiter(int(rpm_limit) if rpm_limit else 50)

        embedding_rpm_limit = os.environ.get("GEMINI_EMBEDDING_RPM_LIMIT")
        self.embedding_rpm_limiter = RateLimiter(
            int(embedding_rpm_limit) if embedding_rpm_limit else 1000
        )

        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set via GEMINI_API_KEY environment variable."
            )

        self._client = genai.Client(api_key=self.api_key)
        logging.info("Using Gemini model: %s", self.model_name)
        logging.info("Using Gemini embedding model: %s", self.embedding_model_name)

    # Main generate method

    @retry(
        retry=retry_if_exception_type(
            (
                google_exceptions.ResourceExhausted,
                google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError,
                google_exceptions.GatewayTimeout,
            )
        ),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=_log_retry_attempt,
    )
    def generate(
        self,
        prompt: str,
        image_path: Union[str, Path],
    ) -> Any:
        """Generate text from an image & prompt via Gemini.

        The method always returns the *raw* text returned by the model. The
        caller is responsible for any downstream parsing/validation.
        """
        self.rpm_limiter()

        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found at {image_path}")

        try:
            with PIL.Image.open(image_path) as image:
                # Build arguments for the generate_content call
                call_kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "contents": [prompt, image],
                }

                response = self._client.models.generate_content(**call_kwargs)

            text_response: str = response.text.strip()
            # The model sometimes wraps the response in markdown code fences – strip them.
            text_response = re.sub(r"^```[a-z]*\n?", "", text_response)
            text_response = re.sub(r"\n?```$", "", text_response)

            return text_response
        except google_exceptions.GoogleAPIError:
            # Propagate Google API errors so the tenacity retry decorator can
            # handle them properly.
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to generate content with Gemini API model: {exc}"
            ) from exc

    @retry(
        retry=retry_if_exception_type(
            (
                google_exceptions.ResourceExhausted,
                google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError,
                google_exceptions.GatewayTimeout,
            )
        ),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=_log_retry_attempt,
    )
    def generate_structured_from_text(
        self,
        prompt_template: str,
        text_input: str,
        response_schema: Union[Type[BaseModel], list[Type[BaseModel]]],
    ) -> Any:
        """Generate structured data from text via Gemini.

        The method returns parsed Pydantic objects as specified by the
        `response_schema`.
        """
        self.rpm_limiter()
        prompt = prompt_template.format(transcription=text_input)

        try:
            # Build arguments for the generate_content call
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                },
            )
            return response.parsed
        except google_exceptions.FailedPrecondition as exc:
            logging.error(
                "Gemini API failed to follow the response schema. Response text: %s",
                response.text,
            )
            raise RuntimeError(f"Failed to parse Gemini response into schema: {exc}") from exc
        except google_exceptions.GoogleAPIError:
            # Propagate Google API errors so the tenacity retry decorator can
            # handle them properly.
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to generate structured content with Gemini API model: {exc}"
            ) from exc

    @retry(
        retry=retry_if_exception_type(
            (
                google_exceptions.ResourceExhausted,
                google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError,
                google_exceptions.GatewayTimeout,
            )
        ),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=_log_retry_attempt,
    )
    def embed_contents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for *texts* while respecting batch-size limits.

        Google's embedding endpoint currently caps requests to 16 texts per call.
        We therefore chunk the input and make multiple calls when necessary while
        preserving the original ordering of results.
        """

        if not texts:
            return []

        MAX_BATCH_SIZE = 16
        embeddings: list[list[float]] = []

        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i : i + MAX_BATCH_SIZE]

            # Apply rate limit per request.
            self.embedding_rpm_limiter()

            try:
                result = self._client.models.embed_content(
                    model=self.embedding_model_name, contents=batch
                )
                embeddings.extend([e.values for e in result.embeddings])
            except google_exceptions.GoogleAPIError:
                # Propagate Google API errors for tenacity to handle.
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to generate embeddings with Gemini API model: {exc}"
                ) from exc

        return embeddings
