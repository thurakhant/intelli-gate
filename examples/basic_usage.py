import asyncio
import os
from src.providers.openai import (
    ChatProvider, 
    CompletionProvider, 
    ImageProvider, 
    EmbeddingProvider, 
    ModelManager
)
from src.utils.config import Config

async def main():
    # Load configuration
    config = Config()
    
    # Demonstrate Chat Completions (Simple Prompt)
    print("=== Chat Completions (Simple Prompt) ===")
    chat_provider = ChatProvider(
        api_key=config.openai_api_key, 
        model="gpt-4o"
    )
    
    # Generate simple chat response
    simple_chat_response = await chat_provider.generate("Explain AI in simple terms")
    print(f"Response: {simple_chat_response.response}")
    print(f"Model: {simple_chat_response.model}")
    print(f"Total Tokens: {simple_chat_response.total_tokens}")
    print(f"Cost: ${simple_chat_response.cost}")
    print('-' * 50)

    # Demonstrate Chat Completions (Complex Conversation)
    print("\n=== Chat Completions (Complex Conversation) ===")
    complex_chat_response = await chat_provider.generate([
        {"role": "system", "content": "You are a technical assistant"},
        {"role": "user", "content": "Explain the difference between AI and Machine Learning"}
    ])
    print(f"Response: {complex_chat_response.response}")
    print(f"Input Tokens: {complex_chat_response.input_tokens}")
    print(f"Output Tokens: {complex_chat_response.output_tokens}")
    print(f"Cost: ${complex_chat_response.cost}")
    print('-' * 50)

    # Demonstrate Completions API
    print("\n=== Traditional Completions ===")
    completion_provider = CompletionProvider(
        api_key=config.openai_api_key
    )
    
    completion_response = await completion_provider.generate(
        "Once upon a time in the world of technology,",
        max_tokens=100
    )
    print(f"Response: {completion_response.response}")
    print(f"Model: {completion_response.model}")
    print(f"Total Tokens: {completion_response.total_tokens}")
    print(f"Cost: ${completion_response.cost}")
    print('-' * 50)

    # Demonstrate Embeddings
    print("\n=== Text Embeddings ===")
    embedding_provider = EmbeddingProvider(
        api_key=config.openai_api_key
    )
    
    # Single text embedding
    single_embedding_response = await embedding_provider.generate(
        "The future of artificial intelligence"
    )
    print("Single Text Embedding:")
    print(f"Embedding Length: {len(single_embedding_response.response)}")
    print(f"Input Tokens: {single_embedding_response.input_tokens}")
    print(f"Cost: ${single_embedding_response.cost}")
    print('-' * 50)

    # Multiple text embeddings
    multi_embedding_response = await embedding_provider.generate([
        "Machine Learning",
        "Deep Learning",
        "Artificial Intelligence"
    ])
    print("\n=== Multiple Text Embeddings ===")
    print(f"Number of Embeddings: {len(multi_embedding_response.response)}")
    print(f"Input Tokens: {multi_embedding_response.input_tokens}")
    print(f"Cost: ${multi_embedding_response.cost}")
    print('-' * 50)

    # Demonstrate Image Generation
    print("\n=== DALL-E Image Generation ===")
    image_provider = ImageProvider(
        api_key=config.openai_api_key, 
        model="dall-e-3"
    )
    
    # Single Image Generation
    single_image_response = await image_provider.generate(
        "A futuristic cityscape with flying cars and neon lights",
        n=1,  # Number of images
        size="1024x1024"
    )
    print("Single Image Generation:")
    print(f"Image URL: {single_image_response.response}")
    print(f"Image Generation Cost: ${single_image_response.cost}")
    print('-' * 50)

    # Multiple Image Generation
    multi_image_response = await image_provider.generate(
        "A serene landscape with mountains and a lake",
        n=2,  # Generate 2 images
        size="1024x1792"
    )
    print("\n=== Multiple Image Generation ===")
    print(f"Image 1 URL: {multi_image_response.response[0]}")
    print(f"Image 2 URL: {multi_image_response.response[1]}")
    print(f"Total Image Generation Cost: ${multi_image_response.cost}")
    print('-' * 50)

    # Demonstrate Available Models
    print("\n=== Available Chat Models ===")
    active_chat_models = ModelManager.filter_models(
        status='active', 
        model_type='chat'
    )
    print("Active Chat Models:")
    for model in active_chat_models:
        details = ModelManager.get_model_details(model)
        print(f"Model: {model}")
        print(f"  Recommended For: {details.get('recommended_for')}")
        print(f"  Context Window: {details.get('context_window')} tokens")
        print(f"  Input Token Cost: ${details.get('input_token_cost', 0)}/1k")
        print(f"  Output Token Cost: ${details.get('output_token_cost', 0)}/1k")
        print('-' * 30)

    # Optional: Download and save images
    def download_image(url, filename):
        import requests
        response = requests.get(url)
        if response.status_code == 200:
            # Create images directory if it doesn't exist
            os.makedirs('images', exist_ok=True)
            with open(f'images/{filename}', 'wb') as f:
                f.write(response.content)
            print(f"Image saved: images/{filename}")
        else:
            print(f"Failed to download image: {url}")

    # Download generated images
    if isinstance(single_image_response.response, str):
        download_image(single_image_response.response, "futuristic_city.png")
    
    if isinstance(multi_image_response.response, list):
        for i, url in enumerate(multi_image_response.response, 1):
            download_image(url, f"serene_landscape_{i}.png")

if __name__ == "__main__":
    asyncio.run(main())