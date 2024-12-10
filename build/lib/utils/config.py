import os
from dotenv import load_dotenv

class Config:
    def __init__(self, env_file: str = '.env'):
        # Load environment variables
        load_dotenv(env_file)

    def get(self, key: str, default: str = None) -> str:
        """
        Retrieve a configuration value
        
        :param key: Environment variable key
        :param default: Default value if key is not found
        :return: Configuration value
        """
        return os.getenv(key, default)

    @property
    def openai_api_key(self) -> str:
        return self.get('OPENAI_API_KEY')

    @property
    def anthropic_api_key(self) -> str:
        return self.get('ANTHROPIC_API_KEY')