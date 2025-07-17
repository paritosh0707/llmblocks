from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from typing import Any
from llmblocks.components.chatbot.chatbot_config import ChatbotConfig

class LangGraphChatbot:
    def __init__(self, model: Any):
        """Initialize the LangGraph chatbot.
        
        Args:
            model: The language model to use for generating responses
        """
        self.model = model
        self.workflow = self._create_workflow()
        # Remove the checkpointer since we don't need persistence for this simple chatbot
        self.app = self.workflow.compile()

    def _create_workflow(self) -> StateGraph:
        """Create and configure the LangGraph workflow.
        
        Returns:
            StateGraph: The configured workflow graph
        """
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_edge(START, "model")
        workflow.add_node("model", self._call_model)
        return workflow

    def _call_model(self, state: MessagesState) -> dict:
        """Call the model with the current message state.
        
        Args:
            state: The current message state
            
        Returns:
            dict: The model's response wrapped in a messages dict
        """
        response = self.model.invoke(state["messages"])
        return {"messages": response}

