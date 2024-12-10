from .base import BaseOpenAIProvider
from src.providers.base_provider import ModelResponse
from typing import Union, List, Optional

class EmbeddingProvider(BaseOpenAIProvider):
    async def generate(self, 
                       input: Union[str, List[str]], 
                       **kwargs) -> ModelResponse:
        """
        Generate embeddings using OpenAI's Embedding API
        
        :param input: Text or list of texts to embed
        :param kwargs: Additional generation parameters
        :return: Comprehensive embedding response
        """
        try:
            # Use specific embedding model
            model = kwargs.get('model', 'text-embedding-ada-002')
            
            # Prepare generation parameters
            generation_params = {
                "model": model,
                "input": input,
                **kwargs
            }

            # Generate embeddings
            raw_response = self.client.embeddings.create(**generation_params)
            
            # Extract embeddings
            embeddings = [data.embedding for data in raw_response.data]

            # Calculate tokens
            if isinstance(input, str):
                input_texts = [input]
            else:
                input_texts = input

            input_tokens = sum(self._calculate_tokens(text) for text in input_texts)

            # Calculate cost (if applicable)
            pricing = self.get_model_pricing(model)
            input_cost = (input_tokens / 1000) * pricing.get("input_token_cost", 0)

            # Create response object
            return ModelResponse(
                provider="OpenAI",
                model=model,
                prompt=str(input),
                response=embeddings[0] if len(embeddings) == 1 else embeddings,
                input_tokens=input_tokens,
                output_tokens=0,
                total_tokens=input_tokens,
                cost=round(input_cost, 4),
                raw_response=raw_response,
                metadata=kwargs
            )

        except Exception as e:
            raise RuntimeError(f"OpenAI Embedding generation error: {str(e)}")