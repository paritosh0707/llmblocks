class ChatbotConfig:
    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 1000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def to_dict(self):
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }