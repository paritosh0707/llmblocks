from pydantic import BaseModel
from typing import Dict, Any, Optional

class MemoryConfig(BaseModel):
    provider_name: str = "in-memory"  # default to in-memory provider
    config: Optional[Dict[str, Any]] = None

class ChatbotConfig(BaseModel):
    name: str
    description: Optional[str]
    llm: Dict[str, Any]
    memory: Optional[MemoryConfig] = None
    system_prompt: Optional[str]
