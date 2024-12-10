from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseProvider(ABC):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the AI model"""
        pass

    @abstractmethod
    def calculate_tokens(self, text: str) -> int:
        """Calculate tokens for the given text"""
        pass

    @abstractmethod
    def get_pricing(self) -> Dict[str, float]:
        """Retrieve current pricing for the provider"""
        pass

    def validate_api_key(self) -> bool:
        """Validate the API key for the provider"""
        return self.api_key is not None and len(self.api_key) > 0