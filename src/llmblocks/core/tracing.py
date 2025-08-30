"""
Tracing and observability utilities for LLMBlocks.

This module provides tracing capabilities for monitoring and debugging
LLMBlocks applications, including integration with popular observability
platforms.
"""

import asyncio
import json
import time
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from uuid import uuid4
from dataclasses import dataclass, asdict
from enum import Enum

from ..utils.logging import get_logger


class TraceLevel(Enum):
    """Trace level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class TraceEvent:
    """Represents a single trace event."""
    id: str
    name: str
    level: TraceLevel
    timestamp: datetime
    duration_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace event to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['level'] = self.level.value
        return data


class TraceCollector:
    """Collects and manages trace events."""
    
    def __init__(self):
        self.events: List[TraceEvent] = []
        self.active_spans: Dict[str, TraceEvent] = {}
        self.logger = get_logger(self.__class__.__name__)
    
    def start_span(
        self,
        name: str,
        level: TraceLevel = TraceLevel.INFO,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        parent_id: Optional[str] = None
    ) -> str:
        """Start a new trace span."""
        span_id = str(uuid4())
        event = TraceEvent(
            id=span_id,
            name=name,
            level=level,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
            tags=tags or {},
            parent_id=parent_id
        )
        
        self.active_spans[span_id] = event
        self.logger.debug(f"Started span: {name} ({span_id})")
        return span_id
    
    def end_span(
        self,
        span_id: str,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """End a trace span."""
        if span_id not in self.active_spans:
            self.logger.warning(f"Attempted to end unknown span: {span_id}")
            return
        
        event = self.active_spans.pop(span_id)
        end_time = datetime.now(timezone.utc)
        duration = (end_time - event.timestamp).total_seconds() * 1000
        
        event.duration_ms = duration
        event.error = error
        
        if metadata:
            event.metadata.update(metadata)
        
        self.events.append(event)
        
        level = "ERROR" if error else "DEBUG"
        self.logger.log(
            level,
            f"Ended span: {event.name} ({span_id}) - {duration:.2f}ms"
        )
    
    def add_event(
        self,
        name: str,
        level: TraceLevel = TraceLevel.INFO,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        parent_id: Optional[str] = None
    ) -> str:
        """Add a point-in-time trace event."""
        event_id = str(uuid4())
        event = TraceEvent(
            id=event_id,
            name=name,
            level=level,
            timestamp=datetime.now(timezone.utc),
            duration_ms=0.0,
            metadata=metadata or {},
            tags=tags or {},
            parent_id=parent_id
        )
        
        self.events.append(event)
        self.logger.debug(f"Added event: {name} ({event_id})")
        return event_id
    
    def get_events(
        self,
        level: Optional[TraceLevel] = None,
        name_filter: Optional[str] = None
    ) -> List[TraceEvent]:
        """Get filtered trace events."""
        events = self.events
        
        if level:
            events = [e for e in events if e.level == level]
        
        if name_filter:
            events = [e for e in events if name_filter in e.name]
        
        return events
    
    def clear(self) -> None:
        """Clear all trace events."""
        self.events.clear()
        self.active_spans.clear()
        self.logger.debug("Cleared all trace events")
    
    def export_json(self) -> str:
        """Export trace events as JSON."""
        return json.dumps([event.to_dict() for event in self.events], indent=2)
    
    def export_to_file(self, filepath: str) -> None:
        """Export trace events to a file."""
        with open(filepath, 'w') as f:
            f.write(self.export_json())
        self.logger.info(f"Exported {len(self.events)} trace events to {filepath}")


# Global trace collector instance
_global_collector = TraceCollector()


def get_tracer() -> TraceCollector:
    """Get the global trace collector."""
    return _global_collector


def set_tracer(collector: TraceCollector) -> None:
    """Set the global trace collector."""
    global _global_collector
    _global_collector = collector


@contextmanager
def trace_span(
    name: str,
    level: TraceLevel = TraceLevel.INFO,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
    parent_id: Optional[str] = None
):
    """Context manager for tracing a code block."""
    tracer = get_tracer()
    span_id = tracer.start_span(name, level, metadata, tags, parent_id)
    
    try:
        yield span_id
    except Exception as e:
        tracer.end_span(span_id, error=str(e))
        raise
    else:
        tracer.end_span(span_id)


@asynccontextmanager
async def trace_async_span(
    name: str,
    level: TraceLevel = TraceLevel.INFO,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
    parent_id: Optional[str] = None
):
    """Async context manager for tracing a code block."""
    tracer = get_tracer()
    span_id = tracer.start_span(name, level, metadata, tags, parent_id)
    
    try:
        yield span_id
    except Exception as e:
        tracer.end_span(span_id, error=str(e))
        raise
    else:
        tracer.end_span(span_id)


def trace_function(
    name: Optional[str] = None,
    level: TraceLevel = TraceLevel.INFO,
    include_args: bool = False,
    include_result: bool = False
):
    """Decorator for tracing function calls."""
    def decorator(func: Callable) -> Callable:
        func_name = name or f"{func.__module__}.{func.__qualname__}"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                metadata = {}
                if include_args:
                    metadata['args'] = str(args)
                    metadata['kwargs'] = str(kwargs)
                
                async with trace_async_span(func_name, level, metadata):
                    result = await func(*args, **kwargs)
                    
                    if include_result:
                        # Add result to metadata (be careful with large objects)
                        result_str = str(result)
                        if len(result_str) > 1000:
                            result_str = result_str[:1000] + "..."
                        metadata['result'] = result_str
                    
                    return result
            
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                metadata = {}
                if include_args:
                    metadata['args'] = str(args)
                    metadata['kwargs'] = str(kwargs)
                
                with trace_span(func_name, level, metadata):
                    result = func(*args, **kwargs)
                    
                    if include_result:
                        # Add result to metadata (be careful with large objects)
                        result_str = str(result)
                        if len(result_str) > 1000:
                            result_str = result_str[:1000] + "..."
                        metadata['result'] = result_str
                    
                    return result
            
            return sync_wrapper
    
    return decorator


def trace_event(
    name: str,
    level: TraceLevel = TraceLevel.INFO,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None
) -> str:
    """Add a point-in-time trace event."""
    return get_tracer().add_event(name, level, metadata, tags)


def trace_error(
    name: str,
    error: Exception,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None
) -> str:
    """Add an error trace event."""
    error_metadata = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        **(metadata or {})
    }
    
    return get_tracer().add_event(
        name,
        TraceLevel.ERROR,
        error_metadata,
        tags
    )


# Integration with popular observability platforms
class LangSmithTracer:
    """Integration with LangSmith tracing."""
    
    def __init__(self, api_key: Optional[str] = None, project: Optional[str] = None):
        self.api_key = api_key
        self.project = project
        self.enabled = bool(api_key)
        
        if self.enabled:
            try:
                import langsmith
                self.client = langsmith.Client(api_key=api_key)
            except ImportError:
                self.enabled = False
                get_logger(__name__).warning(
                    "LangSmith not available. Install with: pip install langsmith"
                )
    
    def export_traces(self, events: List[TraceEvent]) -> None:
        """Export traces to LangSmith."""
        if not self.enabled:
            return
        
        # Implementation would depend on LangSmith API
        # This is a placeholder for future implementation
        pass


class OpenTelemetryTracer:
    """Integration with OpenTelemetry."""
    
    def __init__(self):
        self.enabled = False
        try:
            from opentelemetry import trace
            self.tracer = trace.get_tracer(__name__)
            self.enabled = True
        except ImportError:
            get_logger(__name__).warning(
                "OpenTelemetry not available. Install with: pip install opentelemetry-api"
            )
    
    def export_traces(self, events: List[TraceEvent]) -> None:
        """Export traces to OpenTelemetry."""
        if not self.enabled:
            return
        
        # Implementation would depend on OpenTelemetry setup
        # This is a placeholder for future implementation
        pass


# Convenience functions for common tracing patterns
def trace_llm_call(
    provider: str,
    model: str,
    prompt: str,
    response: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Trace an LLM API call."""
    trace_metadata = {
        'provider': provider,
        'model': model,
        'prompt_length': len(prompt),
        'response_length': len(response),
        **(metadata or {})
    }
    
    return trace_event(
        f"llm_call_{provider}",
        TraceLevel.INFO,
        trace_metadata,
        {'component': 'llm_provider', 'provider': provider}
    )


def trace_block_lifecycle(
    block_id: str,
    block_type: str,
    action: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Trace block lifecycle events."""
    trace_metadata = {
        'block_id': block_id,
        'block_type': block_type,
        'action': action,
        **(metadata or {})
    }
    
    return trace_event(
        f"block_{action}",
        TraceLevel.INFO,
        trace_metadata,
        {'component': 'block_lifecycle', 'block_type': block_type}
    )
