from openai import OpenAI
import tiktoken
from typing import Dict, Any, Literal, Optional
from enum import Enum

class OpenAIModelType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    IMAGE = "image"
    EMBEDDING = "embedding"

class OpenAIRequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    IMAGE = "image"

class BaseOpenAIProvider:
    
    PRICING = {
        
    "dall-e-3": {
        "type": "image",
        "resolution_pricing": {
            "1024x1024": 0.04,   # Cost per image
            "1024x1792": 0.08,   # Rectangular image cost
            "1792x1024": 0.08    # Rectangular image cost
        },
        "status": "active",
        "release_date": "2023-11-06",
        "recommended_for": ["high-quality image generation"]
    },
    "dall-e-2": {
        "type": "image",
        "resolution_pricing": {
            "256x256": 0.016,    # Cost per image
            "512x512": 0.018,    # Cost per image
            "1024x1024": 0.020   # Cost per image
        },
        "status": "active",
        "release_date": "2022-07-15",
        "recommended_for": ["standard image generation"]
    },
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
                 model: Optional[str] = None,
                 request_type: Optional[str] = None):
        """
        Initialize OpenAI Provider with request type selection
        
        :param api_key: OpenAI API key
        :param model: Specific model
        :param request_type: Type of request (chat or completion)
        """
        self.api_key = api_key
        self.model = model or self.get_latest_model()
        self.client = OpenAI(api_key=api_key)
        
        # Set default request type if not provided
        self.request_type = request_type or OpenAIRequestType.CHAT.value
        
        # Initialize encoding
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def _calculate_tokens(self, text: str) -> int:
        """
        Calculate tokens for a given text
        
        :param text: Input text
        :return: Number of tokens
        """
        return len(self.encoding.encode(text))

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

    def get_model_pricing(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve pricing details for a specific model
        
        :param model: Model name (defaults to current model)
        :return: Pricing details dictionary
        """
        model = model or self.model
        return self.PRICING.get(model, {
            "input_token_cost": 0,
            "output_token_cost": 0,
            "context_window": 0
        })