"""
Logging system for LLMBlocks.

This module provides a comprehensive logging system using structlog
with structured logging, performance metrics, and configurable output formats.
"""

import sys
import logging
import json
from typing import Any, Dict, Optional, Union
from datetime import datetime
from pathlib import Path

import structlog
from structlog.stdlib import LoggerFactory
from structlog.processors import (
    TimeStamper,
    JSONRenderer,
    format_exc_info,
    add_log_level,
    StackInfoRenderer,
)
from structlog.typing import FilteringBoundLogger


# Configure structlog to work with standard library logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        TimeStamper(fmt="iso"),
        structlog.processors.UnicodeDecoder(),
        structlog.processors.ExceptionPrettyPrinter(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


class LLMBlocksLogger:
    """
    Enhanced logger for LLMBlocks with additional context and metrics.
    
    This logger provides:
    - Structured logging with JSON output
    - Performance metrics and timing
    - Request/response correlation
    - Error tracking and reporting
    - Configurable output formats
    """
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        output_format: str = "json",
        log_file: Optional[Union[str, Path]] = None,
        enable_metrics: bool = True
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name (usually module or component name)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            output_format: Output format (json, console, both)
            log_file: Optional log file path
            enable_metrics: Whether to enable performance metrics
        """
        self.name = name
        self.level = level.upper()
        self.output_format = output_format
        self.log_file = Path(log_file) if log_file else None
        self.enable_metrics = enable_metrics
        
        # Create the underlying logger
        self._logger = structlog.get_logger(name)
        
        # Set up file logging if specified
        if self.log_file:
            self._setup_file_logging()
        
        # Set up console logging
        self._setup_console_logging()
        
        # Performance tracking
        self._timers: Dict[str, float] = {}
        self._counters: Dict[str, int] = {}
        
        # Log the logger initialization
        self.info(
            "Logger initialized",
            logger_name=name,
            level=level,
            output_format=output_format,
            log_file=str(log_file) if log_file else None
        )
    
    def _setup_file_logging(self) -> None:
        """Set up file logging."""
        if not self.log_file:
            return
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(getattr(logging, self.level))
        
        # Create formatter
        if self.output_format == "json":
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger = logging.getLogger(self.name)
        logger.addHandler(file_handler)
        logger.setLevel(getattr(logging, self.level))
    
    def _setup_console_logging(self) -> None:
        """Set up console logging."""
        # Get the standard library logger
        logger = logging.getLogger(self.name)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.level))
        
        # Create formatter
        if self.output_format == "json":
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        logger.setLevel(getattr(logging, self.level))
    
    def _add_context(self, **kwargs) -> Dict[str, Any]:
        """Add common context to log messages."""
        context = {
            "timestamp": datetime.utcnow().isoformat(),
            "logger": self.name,
            **kwargs
        }
        
        # Add performance metrics if enabled
        if self.enable_metrics:
            context.update({
                "metrics": {
                    "timers": self._timers.copy(),
                    "counters": self._counters.copy()
                }
            })
        
        return context
    
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        context = self._add_context(**kwargs)
        self._logger.debug(message, **context)
    
    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        context = self._add_context(**kwargs)
        self._logger.info(message, **context)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        context = self._add_context(**kwargs)
        self._logger.warning(message, **context)
    
    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        context = self._add_context(**kwargs)
        self._logger.error(message, **context)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message."""
        context = self._add_context(**kwargs)
        self._logger.critical(message, **context)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log an exception message with traceback."""
        context = self._add_context(**kwargs)
        self._logger.exception(message, **context)
    
    # Performance tracking methods
    
    def start_timer(self, name: str) -> None:
        """Start a performance timer."""
        if self.enable_metrics:
            self._timers[name] = datetime.utcnow().timestamp()
    
    def stop_timer(self, name: str) -> float:
        """Stop a performance timer and return duration."""
        if not self.enable_metrics or name not in self._timers:
            return 0.0
        
        start_time = self._timers.pop(name)
        duration = datetime.utcnow().timestamp() - start_time
        
        # Log the timing
        self.debug(f"Timer '{name}' completed", timer_name=name, duration=duration)
        
        return duration
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        if self.enable_metrics:
            self._counters[name] = self._counters.get(name, 0) + value
    
    def set_counter(self, name: str, value: int) -> None:
        """Set a counter to a specific value."""
        if self.enable_metrics:
            self._counters[name] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            "timers": self._timers.copy(),
            "counters": self._counters.copy(),
            "logger_name": self.name
        }
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self._timers.clear()
        self._counters.clear()
        self.debug("Performance metrics reset")
    
    # Context manager for timing operations
    
    def time_operation(self, name: str):
        """Context manager for timing operations."""
        return TimerContext(self, name)
    
    # Request/response correlation
    
    def log_request(self, request_id: str, **kwargs) -> None:
        """Log a request with correlation ID."""
        self.info(
            "Request received",
            request_id=request_id,
            event_type="request",
            **kwargs
        )
    
    def log_response(self, request_id: str, **kwargs) -> None:
        """Log a response with correlation ID."""
        self.info(
            "Response sent",
            request_id=request_id,
            event_type="response",
            **kwargs
        )
    
    def log_error(self, request_id: str, error: Exception, **kwargs) -> None:
        """Log an error with correlation ID."""
        self.error(
            "Error occurred",
            request_id=request_id,
            event_type="error",
            error_type=type(error).__name__,
            error_message=str(error),
            **kwargs
        )


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, logger: LLMBlocksLogger, name: str):
        self.logger = logger
        self.name = name
    
    def __enter__(self):
        self.logger.start_timer(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = self.logger.stop_timer(self.name)
        if exc_type:
            self.logger.error(
                f"Operation '{self.name}' failed",
                operation_name=self.name,
                duration=duration,
                error_type=exc_type.__name__,
                error_message=str(exc_val)
            )


# Global logger instance
_default_logger: Optional[LLMBlocksLogger] = None


def get_logger(
    name: Optional[str] = None,
    level: str = "INFO",
    output_format: str = "json",
    log_file: Optional[Union[str, Path]] = None,
    enable_metrics: bool = True
) -> LLMBlocksLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (defaults to calling module name)
        level: Logging level
        output_format: Output format
        log_file: Optional log file path
        enable_metrics: Whether to enable performance metrics
        
    Returns:
        Configured logger instance
    """
    global _default_logger
    
    if name is None:
        # Get the calling module name
        import inspect
        frame = inspect.currentframe()
        try:
            # Go up the call stack to find the calling module
            while frame and frame.f_back:
                frame = frame.f_back
                if frame.f_globals.get('__name__') != __name__:
                    name = frame.f_globals.get('__name__', 'unknown')
                    break
        finally:
            del frame
    
    if name is None:
        name = 'llmblocks'
    
    # Create new logger instance
    return LLMBlocksLogger(
        name=name,
        level=level,
        output_format=output_format,
        log_file=log_file,
        enable_metrics=enable_metrics
    )


def setup_logging(
    level: str = "INFO",
    output_format: str = "json",
    log_file: Optional[Union[str, Path]] = None,
    enable_metrics: bool = True
) -> None:
    """
    Set up global logging configuration.
    
    Args:
        level: Global logging level
        output_format: Global output format
        log_file: Global log file path
        enable_metrics: Global metrics setting
    """
    global _default_logger
    
    _default_logger = LLMBlocksLogger(
        name="llmblocks",
        level=level,
        output_format=output_format,
        log_file=log_file,
        enable_metrics=enable_metrics
    )


def get_default_logger() -> Optional[LLMBlocksLogger]:
    """Get the default logger instance."""
    return _default_logger


# Convenience functions for common logging patterns

def log_function_call(func_name: str, **kwargs):
    """Decorator to log function calls with parameters."""
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            logger = get_logger()
            logger.debug(
                f"Function '{func_name}' called",
                function_name=func_name,
                args_count=len(args),
                kwargs_count=len(func_kwargs),
                **kwargs
            )
            
            try:
                result = func(*args, **func_kwargs)
                logger.debug(
                    f"Function '{func_name}' completed successfully",
                    function_name=func_name
                )
                return result
            except Exception as e:
                logger.exception(
                    f"Function '{func_name}' failed",
                    function_name=func_name,
                    error=str(e)
                )
                raise
        
        return wrapper
    return decorator


def log_performance(operation_name: str):
    """Decorator to log operation performance."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            logger = get_logger()
            with logger.time_operation(operation_name):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            logger = get_logger()
            with logger.time_operation(operation_name):
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Import asyncio for the performance decorator
import asyncio
