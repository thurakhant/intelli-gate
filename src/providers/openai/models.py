from .base import BaseOpenAIProvider
from typing import List, Dict, Any, Optional

class ModelManager(BaseOpenAIProvider):
    @classmethod
    def list_models(cls) -> List[str]:
        """
        List all available models
        
        :return: List of model names
        """
        return list(cls.PRICING.keys())

    @classmethod
    def get_model_details(cls, model: str) -> Dict[str, Any]:
        """
        Retrieve detailed information about a specific model
        
        :param model: Model name
        :return: Model details
        """
        return cls.PRICING.get(model, {})

    @classmethod
    def filter_models(cls, 
                      status: Optional[str] = None, 
                      model_type: Optional[str] = None) -> List[str]:
        """
        Filter models based on status and type
        
        :param status: Model status (e.g., 'active')
        :param model_type: Model type (e.g., 'chat')
        :return: Filtered list of models
        """
        return [
            model for model, details in cls.PRICING.items()
            if (status is None or details.get('status') == status) and
               (model_type is None or details.get('type') == model_type)
        ]