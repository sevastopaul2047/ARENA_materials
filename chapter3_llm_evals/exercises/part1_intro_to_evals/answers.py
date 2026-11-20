
# %% Imports and setup

import io
import os
import random
import sys
import time
import warnings
from pathlib import Path
from pprint import pprint
from collections.abc import Callable
from typing import Literal, TypeAlias

import httpx
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from tabulate import tabulate
from tqdm.auto import tqdm

# Make sure exercises are in the path
chapter = "chapter3_llm_evals"
section = "part1_intro_to_evals"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists()) # Get the root directory for the chapter
section_dir = root_dir / chapter / "exercises" / section
exercises_dir = root_dir / chapter / "exercises"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

print("ARENA setup working!") #check if the setup is working
print("Repository:", root_dir) #print the repository path
print("Section:", section_dir) #print the section path


# %% First free API call - automatic model fallback

 # Import specific error classes from the OpenAI library for handling API errors.
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)

# WHY USE FREE MODEL FALLBACK?
#
# The primary goal is to verify that our OpenRouter API configuration works
# before purchasing any API credits. We want to confirm that our API key,
# client configuration, and request/response handling work end-to-end
# using only models available through OpenRouter's free tier.
#
# However, free models may experience limited availability, provider
# outages, or temporary capacity restrictions. During initial testing,
# NVIDIA Nemotron returned a provider-side HTTP 404 error.
#
# Rather than treating one model's failure as a failure of our entire
# API setup, we try several free models in a predefined order.
# The first model returning usable text is selected, and no further
# requests are made.
#
# WHY NOT JUST USE max_retries?
#
# The OpenAI SDK's max_retries repeats requests to the same model for
# certain retryable errors. It does not normally retry HTTP 404 errors.
# Our fallback instead switches to a different model when one fails.
# We disable automatic retries to keep API attempts predictable.
#
# LIMITATIONS:
# - Fallback cannot resolve invalid API keys or account-wide restrictions.
# - Failed requests may still count toward free-tier request limits.
# - A successful response verifies this API path, not every model.
# - This mechanism is only for initial API testing and general usage.
#   Controlled evaluations must use fixed models to ensure valid comparisons.
#
# All models below use OpenRouter's free endpoints and are tried in order.

FREE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "dots-studio/dots-3-note-preview:free",
    "google/gemma-4-31b-it:free",
]


def get_first_working_response(
    client: OpenAI,
    messages: list[dict[str, str]],
) -> tuple[str, str] | None:
    """
    Send messages to free OpenRouter models, one at a time.

    Returns:
        (model_name, response_text) for the first successful model.
        None if no model produces a usable text response.

    Error handling:
        - Authentication/access errors: stop immediately.
        - Rate limits: try another model (may be account-wide).
        - Provider/server errors: report and try another model.
        - Network timeouts: report and try another model.

    This is a convenience fallback, not a model evaluation.
    """

    for number, model in enumerate(FREE_MODELS, start=1):
        print(f"\n[{number}/{len(FREE_MODELS)}] Trying {model}")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=300,
            )

            # An API request can succeed without providing usable text.
            if not response.choices:
                print("  No response choices returned. Trying next.")
                continue

            content = response.choices[0].message.content

            if not content or not content.strip():
                print("  Empty text response. Trying next.")
                continue

            # Success: return immediately; remaining models are not called.
            print(f"  Success! Model: {model}")
            pprint(response.model_dump())
            return model, content.strip()

        except AuthenticationError:
            print("  Authentication failed (401). Check OPENROUTER_API_KEY.")
            return None

        except PermissionDeniedError:
            print("  Access denied (403). Check account permissions.")
            return None

        except RateLimitError:
            print("  Rate limit reached (429). Trying next model.")
            print("  Note: account-wide limits affect all models.")

        except APITimeoutError:
            print("  Request timed out. Trying next model.")

        except APIConnectionError:
            print("  Network connection failed. Trying next model.")

        except APIStatusError as error:
            # Don't dump the full provider error: it can contain IDs.
            status = error.status_code

            if status == 404:
                reason = "Model unavailable or provider deployment not found."
            elif status >= 500:
                reason = "Temporary server/provider error."
            elif status == 400:
                reason = "Request rejected; check model parameters."
            else:
                reason = "API request failed."

            print(f"  HTTP {status}: {reason}")
            print("  Trying next model.")

    print("\nNo model returned a usable response.")
    print("Check OpenRouter model availability, quota, and connection.")
    return None


# %% Run the first API experiment

if MAIN:
    # Load the .env file explicitly from the ARENA repository root.
    load_dotenv(root_dir / ".env")

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print("Missing OPENROUTER_API_KEY in ARENA_materials/.env")

    else:
        # max_retries=0 prevents the SDK from making hidden retries.
        # Our function handles fallback to another model instead.
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=30.0,
            max_retries=0,
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        result = get_first_working_response(
            client=openrouter_client,
            messages=messages,
        )

        if result is not None:
            model_name, answer = result
            print(f"\nAnswer from {model_name}:\n{answer}")
