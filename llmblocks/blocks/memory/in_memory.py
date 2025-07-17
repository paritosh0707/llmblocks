from typing import List, Dict, Any
from llmblocks.blocks.memory.base import BaseMemoryProvider

class InMemoryProvider(BaseMemoryProvider):
    def _initialize(self, **kwargs) -> None:
        """Initialize an empty message list."""
        self.messages: List[Dict[str, Any]] = []

    def add(self, message: Dict[str, Any]) -> None:
        """Add a message to memory."""
        self.messages.append(message)

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages from memory."""
        return self.messages

    def clear(self) -> None:
        """Clear all messages from memory."""
        self.messages = [] 