from openai import OpenAI
import tiktoken
from typing import Dict, Any, Optional, List
from ..base_provider import BaseProvider, ModelResponse

class OpenAIProvider(BaseProvider):
    # Comprehensive and up-to-date model pricing and details
    PRICING = {
    # GPT-3.5 Models
    "gpt-3.5-turbo": {
        "input_token_cost": 0.0015,   # per 1000 tokens
        "output_token_cost": 0.002,   # per 1000 tokens
        "context_window": 4096,
        "status": "active",
        "release_date": "2023-03-01",
        "type": "chat",
        "recommended_for": ["general purpose", "cost-effective"]
    },
    "gpt-3.5-turbo-16k": {
        "input_token_cost": 0.003,    # per 1000 tokens
        "output_token_cost": 0.004,   # per 1000 tokens
        "context_window": 16384,
        "status": "active",
        "release_date": "2023-06-15",
        "type": "chat",
        "recommended_for": ["longer context", "detailed analysis"]
    },
    
    # GPT-4 Models
    "gpt-4": {
        "input_token_cost": 0.03,     # per 1000 tokens
        "output_token_cost": 0.06,    # per 1000 tokens
        "context_window": 8192,
        "status": "active",
        "release_date": "2023-03-14",
        "type": "chat",
        "recommended_for": ["complex tasks", "high-quality output"]
    },
    "gpt-4-32k": {
        "input_token_cost": 0.06,     # per 1000 tokens
        "output_token_cost": 0.12,    # per 1000 tokens
        "context_window": 32768,
        "status": "active",
        "release_date": "2023-03-14",
        "type": "chat",
        "recommended_for": ["very long context", "detailed analysis"]
    },
    "gpt-4-turbo": {
        "input_token_cost": 0.01,     # per 1000 tokens
        "output_token_cost": 0.03,    # per 1000 tokens
        "context_window": 128000,
        "status": "active",
        "release_date": "2024-02-15",
        "type": "chat",
        "recommended_for": ["advanced reasoning", "large context"]
    },
    # Latest Models
    "gpt-4o": {
        "input_token_cost": 0.0025,    # per 1000 tokens
        "output_token_cost": 0.01,     # per 1000 tokens
        "context_window": 128000,
        "status": "active",
        "release_date": "2024-05-13",
        "type": "chat",
        "recommended_for": ["multimodal", "high performance", "cost-effective"]
    },
    "gpt-4o-mini": {
        "input_token_cost": 0.00015,   # per 1000 tokens
        "output_token_cost": 0.0006,   # per 1000 tokens
        "context_window": 128000,
        "status": "active",
        "release_date": "2024-07-01",
        "type": "chat",
        "recommended_for": ["lightweight tasks", "cost optimization"]
    }
    }


    def __init__(self, 
                 api_key: str, 
                 model: Optional[str] = None):
        """
        OpenAI Provider with dynamic model selection
        
        :param api_key: OpenAI API key
        :param model: Specific OpenAI model (defaults to latest)
        """
        # Use latest model if not specified
        if model is None:
            model = self.get_latest_model()
        
        super().__init__(api_key, model)
        self.client = OpenAI(api_key=api_key)
        
        # Use tiktoken for the specific model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except Exception:
            # Fallback to default encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")

    @classmethod
    def get_latest_model(cls) -> str:
        """
        Dynamically find the latest active model
        
        :return: Latest model name
        """
        active_models = [
            model for model, details in cls.PRICING.items() 
            if details.get('status') == 'active'
        ]
        
        # Sort by release date and return the most recent
        return max(
            active_models, 
            key=lambda m: cls.PRICING[m].get('release_date', '2000-01-01')
        )

    async def generate(self, 
                       prompt: str, 
                       **kwargs) -> ModelResponse:
        """
        Generate response using OpenAI's API
        
        :param prompt: Input text prompt
        :param kwargs: Additional generation parameters
        :return: Comprehensive model response
        """
        try:
            # Prepare generation parameters
            generation_params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                **kwargs
            }

            # Generate response
            raw_response = self.client.chat.completions.create(**generation_params)
            generated_text = raw_response.choices[0].message.content

            # Calculate tokens
            input_tokens = len(self.encoding.encode(prompt))
            output_tokens = len(self.encoding.encode(generated_text))
            total_tokens = input_tokens + output_tokens

            # Calculate cost
            pricing = self.PRICING.get(self.model, {})
            input_cost = (input_tokens / 1000) * pricing.get("input_token_cost", 0)
            output_cost = (output_tokens / 1000) * pricing.get("output_token_cost", 0)
            total_cost = round(input_cost + output_cost, 4)

            # Create response object
            return ModelResponse(
                provider="OpenAI",
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
            raise RuntimeError(f"OpenAI generation error: {str(e)}")