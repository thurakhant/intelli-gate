from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ModelResponse:
    """
    Comprehensive AI model response tracking
    """
    provider: str
    model: str
    prompt: str
    response: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseProvider(ABC):
    def __init__(self, 
                 api_key: str, 
                 model: str = "default_model"):
        """
        Base AI Provider with standardized interface
        
        :param api_key: Authentication key for the provider
        :param model: Specific model to use
        """
        self.api_key = api_key
        self.model = model

    @abstractmethod
    async def generate(self, 
                       prompt: str, 
                       **kwargs) -> ModelResponse:
        """
        Abstract method to generate a response
        
        :param prompt: Input text prompt
        :param kwargs: Additional generation parameters
        :return: Comprehensive model response
        """
        pass

    def validate_api_key(self) -> bool:
        """
        Validate the provider's API key
        
        :return: API key validity status
        """
        return self.api_key is not None and len(self.api_key) > 0

    def _calculate_tokens(self, text: str) -> int:
        """
        Default token calculation method
        Can be overridden by specific providers
        
        :param text: Text to tokenize
        :return: Number of tokens
        """
        return len(text.split())  # Simple word-based tokenization