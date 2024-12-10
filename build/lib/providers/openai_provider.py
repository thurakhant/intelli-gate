from openai import AsyncOpenAI

aclient = AsyncOpenAI(api_key=api_key)
from .base_provider import BaseProvider
from typing import Dict, Any

class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        super().__init__(api_key)
        self.model = model

    async def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            response = await aclient.chat.completions.create(model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI generation error: {str(e)}")

    def calculate_tokens(self, text: str) -> int:
        import tiktoken
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(text))

    def get_pricing(self) -> Dict[str, float]:
        return {
            "input_token_cost": 0.0015,   # per 1000 tokens
            "output_token_cost": 0.002    # per 1000 tokens
        }