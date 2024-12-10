
# Intelli-Gate: AI Provider Interaction Platform

Intelli-Gate is a flexible platform designed to simplify and unify interactions with various AI service providers. It offers real-time token tracking, cost management, and dynamic model selection. While Intelli-Gate currently supports OpenAI for token and cost tracking, it’s built to be extensible for future integrations and external API support.

---

## 🚀 Features

### 🎉 Currently Available
- **AI Services Integration**:
  - **OpenAI**: Supports GPT-3.5, GPT-4, GPT-4o.
- **Cost Management**:
  - Real-time token usage tracking.
  - Accurate cost calculations for supported OpenAI models.
- **Flexible Model Selection**:
  - Auto-selection of the latest compatible models.
  - Dynamic discovery of available OpenAI models.

### 🛠️ Future Plans
- **API Endpoints** (Planned):
  - RESTful API endpoints for external applications to access AI services.
  - Token usage and cost reporting via APIs.
  - Multi-provider model selection through API.
- **Additional AI Providers** (Planned):
  - Integration with Anthropic, Nvidia, Hugging Face, Cohere, AI21 Labs.

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- `pip` (Python package manager)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/intelli-gate.git
   cd intelli-gate
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration

Create a `.env` file in the project root and add your API keys for supported providers:

```plaintext
OPENAI_API_KEY=your_openai_api_key
```

---

## 🛠️ Usage

### Generating AI Responses (OpenAI Example)

```python
import asyncio
from src.providers.openai_provider import OpenAIProvider
from src.utils.config import Config

async def main():
    # Load configuration
    config = Config()

    # Initialize OpenAI Provider
    openai_provider = OpenAIProvider(
        api_key=config.openai_api_key,
        model="gpt-4o"
    )

    # Generate a response
    response = await openai_provider.generate("Explain AI")
    print(f"Response: {response.response}")
    print(f"Total Tokens: {response.total_tokens}")
    print(f"Cost: ${response.cost}")

asyncio.run(main())
```

---

## 🌐 Future API Endpoints (Planned)

*Note: API endpoints are currently not implemented but are planned for future releases.*

Planned endpoints include:
- `POST /providers/generate`: Generate AI responses.
- `GET /tokens/total`: Retrieve total token usage.
- `GET /tokens/usage-log`: Retrieve detailed token usage logs.

---

## 🚧 Roadmap

### Current State
- AI services integration (OpenAI only).
- Internal token and cost tracking.

### Future Plans
- Add API endpoints for external integrations.
- Expand support for additional AI providers (Anthropic, Nvidia, Hugging Face, etc.).
- Enhanced reporting features via API.

---

## 💡 About

Intelli-Gate is designed to streamline AI interactions and provide developers with a foundation for cost-efficient operations. As the project evolves, it aims to include robust API endpoints and support for additional AI providers.

---

## 🤝 Contributing

We welcome contributions! Whether you want to report bugs, suggest new features, or contribute code, we encourage you to get involved.

### How to Contribute
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a pull request on GitHub.

### Code of Conduct
Please review our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming environment for everyone.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🔗 Contact

- GitHub: [yourusername](https://github.com/yourusername)
- Email: [youremail@example.com](mailto:youremail@example.com)
