"""
LangGraph Compatibility Utilities.

This module provides compatibility utilities for different LangGraph versions,
handling MRO conflicts and API changes.
"""

import sys
from typing import Any, Dict, List, Optional, Callable
import warnings


def safe_import_langgraph():
    """
    Safely import LangGraph components, handling MRO conflicts.
    
    Returns:
        tuple: (success, components_dict, error_message)
    """
    try:
        # Try to import LangGraph components
        from langgraph.graph import StateGraph, MessagesState, START, END
        from langgraph.checkpoint.memory import MemorySaver
        
        return True, {
            'StateGraph': StateGraph,
            'MessagesState': MessagesState,
            'START': START,
            'END': END,
            'MemorySaver': MemorySaver
        }, None
        
    except TypeError as e:
        if "method resolution order" in str(e).lower():
            # MRO conflict - try alternative approaches
            try:
                # Try importing individual components
                components = {}
                
                # Try MessagesState from different locations
                try:
                    from langgraph.graph.message import MessagesState
                    components['MessagesState'] = MessagesState
                except ImportError:
                    pass
                
                # Try constants
                try:
                    from langgraph.constants import START, END
                    components['START'] = START
                    components['END'] = END
                except ImportError:
                    # Fallback constants
                    components['START'] = "__start__"
                    components['END'] = "__end__"
                
                return False, components, f"MRO conflict: {e}"
                
            except Exception as fallback_error:
                return False, {}, f"MRO conflict and fallback failed: {fallback_error}"
        else:
            return False, {}, f"Import error: {e}"
    
    except ImportError as e:
        return False, {}, f"LangGraph not available: {e}"
    
    except Exception as e:
        return False, {}, f"Unexpected error: {e}"


def create_compatible_graph(llm_node_func: Callable, node_name: str = "llm"):
    """
    Create a LangGraph-compatible graph using the best available API.
    
    Args:
        llm_node_func: The LLM node function
        node_name: Name for the LLM node
    
    Returns:
        tuple: (success, graph_or_none, error_message)
    """
    success, components, error = safe_import_langgraph()
    
    if success and 'StateGraph' in components:
        try:
            # Use full LangGraph API
            graph = components['StateGraph'](components['MessagesState'])
            graph.add_node(node_name, llm_node_func)
            graph.add_edge(components['START'], node_name)
            graph.add_edge(node_name, components['END'])
            
            compiled_graph = graph.compile()
            return True, compiled_graph, None
            
        except Exception as e:
            return False, None, f"Graph creation failed: {e}"
    
    else:
        # Fallback: Create a simple callable that mimics graph behavior
        async def simple_graph(initial_state: Dict[str, Any]) -> Dict[str, Any]:
            """Simple graph fallback that just calls the LLM node."""
            return await llm_node_func(initial_state)
        
        return False, simple_graph, f"Using fallback graph: {error}"


def get_langgraph_version():
    """Get the LangGraph version if available."""
    try:
        import langgraph
        return getattr(langgraph, '__version__', 'unknown')
    except ImportError:
        return None


def check_langgraph_compatibility():
    """
    Check LangGraph compatibility and return a report.
    
    Returns:
        dict: Compatibility report
    """
    version = get_langgraph_version()
    success, components, error = safe_import_langgraph()
    
    report = {
        'version': version,
        'available': version is not None,
        'import_success': success,
        'components_available': list(components.keys()) if components else [],
        'error': error,
        'mro_conflict': 'method resolution order' in str(error).lower() if error else False,
        'recommended_action': None
    }
    
    if not report['available']:
        report['recommended_action'] = "Install LangGraph: pip install langgraph"
    elif report['mro_conflict']:
        report['recommended_action'] = "Using MRO-safe fallback implementation"
    elif not success:
        report['recommended_action'] = f"Check LangGraph installation: {error}"
    else:
        report['recommended_action'] = "Full LangGraph compatibility available"
    
    return report


# Compatibility warnings
def warn_about_compatibility():
    """Issue warnings about LangGraph compatibility if needed."""
    report = check_langgraph_compatibility()
    
    if report['mro_conflict']:
        warnings.warn(
            "LangGraph MRO conflict detected. Using fallback implementation. "
            "This is a known issue with certain Python/LangGraph version combinations.",
            UserWarning,
            stacklevel=2
        )
    elif not report['import_success'] and report['available']:
        warnings.warn(
            f"LangGraph import issues detected: {report['error']}. "
            "Some features may not be available.",
            UserWarning,
            stacklevel=2
        )
