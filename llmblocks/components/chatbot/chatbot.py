from llmblocks.components.chatbot.chatbot_config import ChatbotConfig, MemoryConfig
from llmblocks.blocks.llm_provider import LLMFactory
from llmblocks.blocks.memory.factory import MemoryFactory
from llmblocks.components.chatbot.langgraph_implementation import LangGraphChatbot
from typing import Optional, Union, List, Dict, Any
import yaml
import uuid

class Chatbot:
    def __init__(
        self, 
        config: Optional[ChatbotConfig] = None,
        config_yaml_path: Optional[str] = None,
    ):
        if config_yaml_path:
            self.config = self._load_config_yaml(config_yaml_path)
        else:
            self.config = config

        if not self.config:
            raise ValueError("No config provided")

        # Generate a unique session ID for this chatbot instance
        self.session_id = uuid.uuid4().hex
        
        # Initialize LLM
        self.llm = LLMFactory.create_provider(**self.config.llm).get_llm()
        
        # Initialize memory
        memory_config = self.config.memory or MemoryConfig()
        self.memory = MemoryFactory.create_provider(
            provider_name=memory_config.provider_name,
            session_id=self.session_id,
            **(memory_config.config or {})
        )
        
        self.system_prompt = self.config.system_prompt
        self._chatbot = self._get_langgraph_chatbot()

    def chat(self, message: str):
        # Create message object
        message_obj = {"role": "user", "content": message}
        
        # Add to memory
        self.memory.add(message_obj)
        
        # Get all messages including history
        messages = self.memory.get_messages()
        
        # Get response from model
        response = self._chatbot.app.invoke({"messages": messages})
        
        # Extract the last message (model's response)
        response_message = response['messages'][-1]
        
        # Add model's response to memory
        self.memory.add(response_message)
        
        return response_message.content

    def _get_system_prompt(self):
        if self.system_prompt:
            return self.system_prompt
        if self.config and self.config.name:
            return f"You are a helpful assistant named {self.config.name}. {self.config.description}"
        return "You are a helpful assistant."

    def _get_langgraph_chatbot(self):
        return LangGraphChatbot(self.llm)

    def _load_config_yaml(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
            return ChatbotConfig.model_validate(config_dict)