from typing import Dict
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class TokenUsageEntry:
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)

class TokenTracker:
    def __init__(self):
        self._total_tokens: int = 0
        self._provider_tokens: Dict[str, int] = {}
        self._usage_log: Dict[str, TokenUsageEntry] = {}

    def track_tokens(self, provider: str, input_tokens: int, output_tokens: int):
        total_tokens = input_tokens + output_tokens
        
        # Update total tokens
        self._total_tokens += total_tokens
        
        # Update provider-specific tokens
        self._provider_tokens[provider] = self._provider_tokens.get(provider, 0) + total_tokens
        
        # Log usage
        entry = TokenUsageEntry(
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens
        )
        self._usage_log[str(entry.timestamp)] = entry

    def get_total_tokens(self) -> int:
        return self._total_tokens

    def get_provider_tokens(self, provider: str = None) -> Dict[str, int]:
        return self._provider_tokens.get(provider, 0) if provider else self._provider_tokens

    def get_usage_log(self) -> Dict[str, TokenUsageEntry]:
        return self._usage_log