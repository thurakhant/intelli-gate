from .base import BaseOpenAIProvider
from src.providers.base_provider import ModelResponse
from typing import Union, List, Optional

class ImageProvider(BaseOpenAIProvider):
    async def generate(self, 
                       prompt: str, 
                       **kwargs) -> ModelResponse:
        """
        Generate images using DALL-E with comprehensive cost tracking
        
        :param prompt: Image generation prompt
        :param kwargs: Additional generation parameters
        :return: Model response with image details and cost
        """
        try:
            # Default generation parameters
            model = kwargs.get('model', 'dall-e-3')
            
            # Enforce model-specific constraints
            if model == 'dall-e-3':
                # DALL-E 3 only supports n=1
                kwargs['n'] = 1
                
                # Validate size for DALL-E 3
                size = kwargs.get('size', '1024x1024')
                valid_sizes = ['1024x1024', '1792x1024', '1024x1792']
                if size not in valid_sizes:
                    size = '1024x1024'
                kwargs['size'] = size

            generation_params = {
                "model": model,
                "prompt": prompt,
                "n": kwargs.get("n", 1),  # Number of images
                "size": kwargs.get("size", "1024x1024"),
                **kwargs
            }

            # Generate images
            response = self.client.images.generate(**generation_params)
            
            # Calculate image generation cost
            pricing = self.get_model_pricing(model)
            resolution_pricing = pricing.get("resolution_pricing", {})
            size = generation_params["size"]
            
            # Calculate total cost based on number of images and resolution
            image_cost = resolution_pricing.get(size, 0) * generation_params["n"]

            # Extract image URLs
            image_urls = [img.url for img in response.data]

            return ModelResponse(
                provider="OpenAI",
                model=model,
                prompt=prompt,
                response=image_urls[0] if len(image_urls) == 1 else image_urls,
                cost=image_cost,
                raw_response=response,
                metadata={
                    "size": size,
                    "num_images": generation_params["n"]
                }
            )

        except Exception as e:
            raise RuntimeError(f"OpenAI Image generation error: {str(e)}")