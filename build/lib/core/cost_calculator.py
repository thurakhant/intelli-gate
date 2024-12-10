from typing import Dict

class CostCalculator:
    PROVIDER_PRICING = {
        "openai": {
            "input_cost_per_1k": 0.0015,
            "output_cost_per_1k": 0.002
        },
        "anthropic": {
            "input_cost_per_1k": 0.003,
            "output_cost_per_1k": 0.004
        },
        # Add more providers as needed
    }

    @classmethod
    def calculate_cost(cls, provider: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost based on provider and token usage
        
        :param provider: Name of the AI provider
        :param input_tokens: Number of input tokens
        :param output_tokens: Number of output tokens
        :return: Total cost in USD
        """
        provider_lower = provider.lower()
        
        if provider_lower not in cls.PROVIDER_PRICING:
            raise ValueError(f"No pricing information for provider: {provider}")

        pricing = cls.PROVIDER_PRICING[provider_lower]
        
        input_cost = (input_tokens / 1000) * pricing["input_cost_per_1k"]
        output_cost = (output_tokens / 1000) * pricing["output_cost_per_1k"]
        
        return round(input_cost + output_cost, 4)

    @classmethod
    def add_provider_pricing(cls, provider: str, input_cost: float, output_cost: float):
        """
        Dynamically add or update pricing for a provider
        
        :param provider: Name of the provider
        :param input_cost: Cost per 1000 input tokens
        :param output_cost: Cost per 1000 output tokens
        """
        cls.PROVIDER_PRICING[provider.lower()] = {
            "input_cost_per_1k": input_cost,
            "output_cost_per_1k": output_cost
        }