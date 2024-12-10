from .base import BaseOpenAIProvider
from src.providers.base_provider import ModelResponse
from typing import Union, Optional

class CompletionProvider(BaseOpenAIProvider):
    async def generate(self, 
                       prompt: str, 
                       **kwargs) -> ModelResponse:
        """
        Generate text using OpenAI Completions API
        
        :param prompt: Input text prompt
        :param kwargs: Additional generation parameters
        :return: Comprehensive model response
        """
        try:
            # Use specific completions model
            model = kwargs.get('model', 'gpt-3.5-turbo-instruct')
            
            # Prepare generation parameters
            generation_params = {
                "model": model,
                "prompt": prompt,
                "max_tokens": kwargs.get('max_tokens', 100),
                **kwargs
            }

            # Generate response
            raw_response = self.client.completions.create(**generation_params)
            generated_text = raw_response.choices[0].text.strip()

            # Calculate tokens
            input_tokens = self._calculate_tokens(prompt)
            output_tokens = self._calculate_tokens(generated_text)
            total_tokens = input_tokens + output_tokens

            # Calculate cost
            pricing = self.get_model_pricing(model)
            input_cost = (input_tokens / 1000) * pricing.get("input_token_cost", 0)
            output_cost = (output_tokens / 1000) * pricing.get("output_token_cost", 0)
            total_cost = round(input_cost + output_cost, 4)

            # Create response object
            return ModelResponse(
                provider="OpenAI",
                model=model,
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
            raise RuntimeError(f"OpenAI Completions generation error: {str(e)}")