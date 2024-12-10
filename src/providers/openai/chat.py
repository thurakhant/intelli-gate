from .base import BaseOpenAIProvider, OpenAIRequestType
from src.providers.base_provider import ModelResponse
from typing import Union, List, Dict, Any

class ChatProvider(BaseOpenAIProvider):
    async def generate(self, 
                       prompt: Union[str, List[Dict[str, str]]], 
                       **kwargs) -> ModelResponse:
        """
        Generate response using either Chat or Completion API
        
        :param prompt: User prompt (string or message list)
        :param kwargs: Additional generation parameters
        :return: Model response
        """
        try:
            # Ensure request_type is set, defaulting to chat if not specified
            request_type = kwargs.get('request_type', self.request_type)
            
            # Prepare generation parameters based on request type
            if request_type == OpenAIRequestType.CHAT.value:
                # Chat Completions API
                if isinstance(prompt, str):
                    messages = [{"role": "user", "content": prompt}]
                else:
                    messages = prompt

                generation_params = {
                    "model": self.model,
                    "messages": messages,
                    **kwargs
                }
                raw_response = self.client.chat.completions.create(**generation_params)
                generated_text = raw_response.choices[0].message.content
                
                # Calculate tokens for chat
                input_tokens = sum(
                    self._calculate_tokens(msg.get('content', '')) 
                    for msg in messages
                )
                output_tokens = self._calculate_tokens(generated_text)

            elif request_type == OpenAIRequestType.COMPLETION.value:
                # Traditional Completions API
                generation_params = {
                    "model": self.model,
                    "prompt": prompt if isinstance(prompt, str) else str(prompt),
                    **kwargs
                }
                raw_response = self.client.completions.create(**generation_params)
                generated_text = raw_response.choices[0].text.strip()
                
                # Calculate tokens for completion
                input_tokens = self._calculate_tokens(generation_params["prompt"])
                output_tokens = self._calculate_tokens(generated_text)

            else:
                raise ValueError(f"Unsupported request type: {request_type}")

            # Calculate total tokens and cost
            total_tokens = input_tokens + output_tokens
            pricing = self.get_model_pricing(self.model)
            input_cost = (input_tokens / 1000) * pricing.get("input_token_cost", 0)
            output_cost = (output_tokens / 1000) * pricing.get("output_token_cost", 0)
            total_cost = round(input_cost + output_cost, 4)

            # Create response object
            return ModelResponse(
                provider="OpenAI",
                model=self.model,
                prompt=str(prompt),
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