"""
Vectorless RAG — LLM Service
================================
Wraps the NVIDIA NIM API via the OpenAI Python SDK for all LLM interactions.

Uses the OpenAI SDK pointed at NVIDIA's integration endpoint:
    base_url = "https://integrate.api.nvidia.com/v1"
    model    = "nvidia/nemotron-3-ultra-550b-a55b"

Features:
    - OpenAI SDK client (standard, enterprise-grade interface)
    - Retry logic with exponential backoff
    - Streaming support with reasoning_content extraction
    - Response validation and error classification
    - Structured prompt formatting
    - JSON extraction from LLM responses
    - Thinking/reasoning mode via chat_template_kwargs

This service is used by IndexService (for tree building) and
QueryService (for traversal reasoning and answer synthesis).
"""

import json
import time
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, AuthenticationError, APIStatusError
from flask import current_app
from app.utils.logger import get_logger, log_performance
from app.utils.exceptions import (
    LLMServiceError,
    LLMAuthenticationError,
    LLMRateLimitError,
)

logger = get_logger(__name__)


class LLMService:
    """
    Client for the NVIDIA NIM API via OpenAI SDK.

    Provides methods to send prompts to the LLM and receive
    structured or free-form responses. Uses the standard OpenAI
    Python SDK configured to point at NVIDIA's API base URL.
    """

    def __init__(self, app=None):
        """
        Initialize the LLM service.

        Args:
            app: Flask app instance. If None, uses current_app proxy.
        """
        self.app = app
        self._client = None

    def _get_config(self):
        """Get configuration from the Flask app context."""
        app = self.app or current_app
        return {
            "api_key": app.config["NVIDIA_API_KEY"],
            "api_base_url": app.config["NVIDIA_API_BASE_URL"],
            "model": app.config["LLM_MODEL"],
            "max_tokens": app.config["LLM_MAX_TOKENS"],
            "temperature": app.config["LLM_TEMPERATURE"],
            "top_p": app.config["LLM_TOP_P"],
            "max_retries": app.config["MAX_RETRIES"],
            "timeout": app.config["REQUEST_TIMEOUT"],
        }

    def _get_client(self):
        """
        Get or create the OpenAI client configured for NVIDIA NIM.

        The client is lazily initialized and cached for reuse.
        Uses the standard OpenAI SDK layout with the NVIDIA base URL override.
        """
        if self._client is None:
            config = self._get_config()

            if not config["api_key"] or config["api_key"] == "your_nvidia_api_key_here":
                raise LLMAuthenticationError(
                    "NVIDIA API key is not configured. Set NVIDIA_API_KEY in your .env file."
                )

            self._client = OpenAI(
                base_url=config["api_base_url"],
                api_key=config["api_key"],
                timeout=config["timeout"],
            )
            logger.debug(f"OpenAI client initialized for NVIDIA NIM: {config['api_base_url']}")

        return self._client

    @log_performance
    def generate(self, user_prompt, system_prompt=None, max_tokens=None, temperature=None, enable_thinking=False):
        """
        Send a prompt to the NVIDIA NIM API and return the response text.

        Uses streaming to collect the full response, handling both
        regular content and reasoning_content from the model.

        Args:
            user_prompt:      The user message content.
            system_prompt:    Optional system message for role/behavior instructions.
            max_tokens:       Override the default max_tokens.
            temperature:      Override the default temperature.
            enable_thinking:  Enable the model's reasoning/thinking mode.

        Returns:
            The LLM's response text (string).

        Raises:
            LLMServiceError: If the API call fails after all retries.
            LLMAuthenticationError: If the API key is invalid.
            LLMRateLimitError: If rate limited.
        """
        config = self._get_config()
        client = self._get_client()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Build kwargs for the API call
        call_kwargs = {
            "model": config["model"],
            "messages": messages,
            "max_tokens": max_tokens or config["max_tokens"],
            "temperature": temperature if temperature is not None else config["temperature"],
            "top_p": config["top_p"],
            "stream": True,
        }

        # Enable thinking/reasoning mode if requested
        if enable_thinking:
            call_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True}
            }

        # Execute with retry logic
        return self._execute_with_retry(client, call_kwargs, config["max_retries"])

    @log_performance
    def generate_json(self, user_prompt, system_prompt=None, max_tokens=None):
        """
        Send a prompt expecting a JSON response from the LLM.

        Wraps generate() and extracts valid JSON from the response.

        Args:
            user_prompt:   The user prompt (should instruct JSON output).
            system_prompt: Optional system prompt.
            max_tokens:    Override max_tokens.

        Returns:
            Parsed JSON object (dict or list).

        Raises:
            LLMServiceError: If JSON parsing fails or API errors.
        """
        # Add JSON instruction to system prompt
        json_instruction = (
            "You MUST respond with valid JSON only. "
            "Do not include any text before or after the JSON. "
            "Do not wrap the JSON in markdown code blocks."
        )
        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{json_instruction}"
        else:
            system_prompt = json_instruction

        # Use lower temperature for structured output
        response_text = self.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            enable_thinking=False,  # Thinking mode may interfere with JSON output
        )

        # Parse JSON from response
        return self._extract_json(response_text)

    def _execute_with_retry(self, client, call_kwargs, max_retries):
        """
        Execute a streaming API call with exponential backoff retry logic.

        Collects the streamed response, handling both regular content
        and reasoning_content from NVIDIA models.

        Args:
            client:      The OpenAI client instance.
            call_kwargs: Arguments for the chat.completions.create call.
            max_retries: Maximum number of retry attempts.

        Returns:
            The full LLM response text.

        Raises:
            LLMServiceError: After all retries are exhausted.
        """
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(f"LLM API call attempt {attempt}/{max_retries}")

                # Make the streaming API call
                completion = client.chat.completions.create(**call_kwargs)

                # Collect the streamed response
                content_parts = []
                reasoning_parts = []

                for chunk in completion:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    # Capture reasoning content (thinking mode)
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        reasoning_parts.append(reasoning)

                    # Capture regular content
                    if delta.content is not None:
                        content_parts.append(delta.content)

                full_content = "".join(content_parts).strip()

                if reasoning_parts:
                    reasoning_text = "".join(reasoning_parts)
                    logger.debug(f"Model reasoning ({len(reasoning_text)} chars captured)")

                if not full_content:
                    raise LLMServiceError("NVIDIA API returned an empty response.")

                logger.debug(f"Response received: {len(full_content)} chars")
                return full_content

            except AuthenticationError as e:
                raise LLMAuthenticationError(
                    f"NVIDIA API authentication failed. Check your API key. Detail: {e}"
                )

            except RateLimitError as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Rate limited. Waiting {wait_time}s before retry... ({e})"
                    )
                    time.sleep(wait_time)
                    last_error = e
                    continue
                raise LLMRateLimitError(
                    f"NVIDIA API rate limit exceeded after {max_retries} attempts."
                )

            except (APIConnectionError, APITimeoutError) as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Network error: {e}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise LLMServiceError(
                        f"Failed to connect to NVIDIA API after {max_retries} attempts: {e}"
                    )

            except APIStatusError as e:
                if e.status_code >= 500 and attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Server error {e.status_code}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    last_error = e
                    continue
                raise LLMServiceError(
                    f"NVIDIA API error ({e.status_code}): {e.message}"
                )

            except (LLMAuthenticationError, LLMRateLimitError, LLMServiceError):
                raise

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error during LLM call: {e}", exc_info=True)
                if attempt >= max_retries:
                    raise LLMServiceError(
                        f"Unexpected error after {max_retries} attempts: {e}"
                    )
                wait_time = 2 ** attempt
                time.sleep(wait_time)

        raise LLMServiceError(f"All {max_retries} retry attempts failed: {last_error}")

    def _extract_json(self, text):
        """
        Extract and parse JSON from LLM response text.

        Handles common LLM quirks:
        - JSON wrapped in markdown code blocks (```json ... ```)
        - Leading/trailing whitespace
        - Multiple JSON objects (takes the first valid one)

        Args:
            text: The raw LLM response text.

        Returns:
            Parsed JSON object (dict or list).

        Raises:
            LLMServiceError: If no valid JSON can be extracted.
        """
        # Strip whitespace
        text = text.strip()

        # Remove markdown code block wrappers if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object within the text
        # Look for the first { ... } or [ ... ]
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start_idx = text.find(start_char)
            if start_idx != -1:
                # Find matching closing bracket
                depth = 0
                for i in range(start_idx, len(text)):
                    if text[i] == start_char:
                        depth += 1
                    elif text[i] == end_char:
                        depth -= 1
                    if depth == 0:
                        json_str = text[start_idx:i + 1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            break

        raise LLMServiceError(
            f"Failed to extract valid JSON from LLM response. "
            f"Response preview: {text[:200]}..."
        )

    def health_check(self):
        """
        Verify connectivity to the NVIDIA NIM API.

        Sends a minimal prompt to confirm the API key is valid
        and the model is accessible.

        Returns:
            dict with status and latency information.
        """
        try:
            start = time.time()
            response = self.generate(
                user_prompt="Respond with exactly one word: OK",
                system_prompt="You are a health check responder. Reply with only 'OK'. Be concise.",
                max_tokens=10,
                temperature=0.0,
                enable_thinking=False,
            )
            latency_ms = round((time.time() - start) * 1000, 2)

            return {
                "status": "healthy",
                "latency_ms": latency_ms,
                "model": self._get_config()["model"],
                "response": response[:50],
            }
        except LLMAuthenticationError:
            return {"status": "auth_error", "message": "API key is invalid or not set."}
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}
