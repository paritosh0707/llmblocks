import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from llmblocks.components.chatbot.chatbot import Chatbot

def main():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create a chatbot instance using the config file
    config_path = os.path.join(current_dir, "chatbot_config.yaml")
    chatbot = Chatbot(config_yaml_path=config_path)
    
    print("Chatbot initialized! Type 'quit' to exit.")
    print("-" * 50)
    
    # Simple chat loop
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for exit command
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        # Get chatbot response
        try:
            response = chatbot.chat(user_input)
            print("\nChatbot:", response)
            print("-" * 50)
        except Exception as e:
            print(f"Error: {str(e)}")
            print("-" * 50)

if __name__ == "__main__":
    main()
