import anthropic
from .base_provider import BaseProvider, ModelResponse

class AnthropicProvider(BaseProvider):
    PRICING = {
        "claude-2": {
            "input_token_cost": 0.008,    # per 1000 tokens
            "output_token_cost": 0.024    # per 1000 tokens
        }
    }

    def __init__(self, 
                 api_key: str, 
                 model: str = "claude-2"):
        """
        Anthropic Provider with Claude models
        
        :param api_key: Anthropic API key
        :param model: Specific Anthropic model
        """
        super().__init__(api_key, model)
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, 
                 prompt: str, 
                 **kwargs) -> ModelResponse:
        """
        Generate response using Anthropic's API
        
        :param prompt: Input text prompt
        :param kwargs: Additional Anthropic generation parameters
        :return: Comprehensive model response
        """
        try:
            # Prepare generation parameters
            generation_params = {
                "model": self.model,
                "prompt": prompt,
                **kwargs
            }

            # Generate response
            raw_response = self.client.completions.create(**generation_params)
            generated_text = raw_response.completion

            # Calculate tokens
            input_tokens = self.client.count_tokens(prompt)
            output_tokens = self.client.count_tokens(generated_text)
            total_tokens = input_tokens + output_tokens

            # Calculate cost
            pricing = self.PRICING.get(self.model, {})
            input_cost = (input_tokens / 1000) * pricing.get("input_token_cost", 0)
            output_cost = (output_tokens / 1000) * pricing.get("output_token_cost", 0)
            total_cost = round(input_cost + output_cost, 4)

            # Create response object
            return ModelResponse(
                provider="Anthropic",
                model=self.model,
                prompt=prompt,
                response=generated_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost=total_cost,
                raw_response=raw_response,
                metadata=kwargs
            )

        except Exception as e:
            raise RuntimeError(f"Anthropic generation error: {str(e)}")