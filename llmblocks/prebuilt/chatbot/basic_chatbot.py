from llmblocks.prebuilt import ChatbotConfig


class BasicChatbot:
    def __init__(self, config: ChatbotConfig):
        self.config = config

    def chat(self, message: str) -> str:
        return "Hello, world!"