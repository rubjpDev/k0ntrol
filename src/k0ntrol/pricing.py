"""Estimated model prices, for the session cost in the status bar.

USD per 1M tokens (input, output) for the models the setup screen offers.
Bedrock on-demand, US East, standard tier, read from the Amazon Bedrock
pricing page on 2026-11-09. Estimates only: the provider's bill is the
truth. A model without a price here shows tokens but no cost.
"""

PRICES: dict[str, tuple[float, float]] = {
    "qwen.qwen3-next-80b-a3b": (0.15, 1.20),
    "qwen.qwen3-coder-next": (0.50, 1.20),
    "us.amazon.nova-pro-v1:0": (0.80, 3.20),
    "us.amazon.nova-lite-v1:0": (0.06, 0.24),
}

# Models that run on the user's own machine cost nothing per token.
LOCAL_PROVIDERS = frozenset({"ollama", "lmstudio"})


def estimate_cost(provider: str | None, model: str | None, tokens_in: int, tokens_out: int) -> float | None:
    """The call's estimated cost in USD, or None when the price is unknown."""
    if provider in LOCAL_PROVIDERS:
        return 0.0
    price = PRICES.get(model or "")
    if price is None:
        return None
    return (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000
