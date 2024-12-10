import asyncio
from src.providers.openai_provider import OpenAIProvider
from src.providers.anthropic_provider import AnthropicProvider
from src.utils.config import Config

def main():
    # Load configuration
    config = Config()

    # Initialize providers
    openai_provider = OpenAIProvider(
        api_key=config.openai_api_key, 
        model="gpt-3.5-turbo"
    )
    
    anthropic_provider = AnthropicProvider(
        api_key=config.anthropic_api_key,
        model="claude-2"
    )

    # Example prompts
    prompts = [
        "Explain quantum computing in simple terms",
        "Describe the future of artificial intelligence"
    ]

    # Demonstrate provider usage
    for prompt in prompts:
        print(f"\nPrompt: {prompt}\n")
        
        # OpenAI Response
        openai_response = openai_provider.generate(prompt)
        print("OpenAI Response:")
        print(f"Generated Text: {openai_response.response}")
        print(f"Total Cost: ${openai_response.cost}")
        print(f"Total Tokens: {openai_response.total_tokens}")

        # Anthropic Response
        anthropic_response = anthropic_provider.generate(prompt)
        print("\nAnthropic Response:")
        print(f"Generated Text: {anthropic_response.response}")
        print(f"Total Cost: ${anthropic_response.cost}")
        print(f"Total Tokens: {anthropic_response.total_tokens}")

if __name__ == "__main__":
    main()