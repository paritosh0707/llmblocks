"""
Core utilities for LLMBlocks.

This module provides common utility functions used across the framework.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import yaml


def generate_id(prefix: str = "") -> str:
    """Generate a unique identifier with optional prefix."""
    unique_id = str(uuid4())
    return f"{prefix}_{unique_id}" if prefix else unique_id


def get_timestamp() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format datetime as ISO string."""
    if dt is None:
        dt = get_timestamp()
    return dt.isoformat()


def hash_dict(data: Dict[str, Any]) -> str:
    """Create a hash of a dictionary for caching/comparison."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


def load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file safely."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_yaml_file(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save data to YAML file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file safely."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save data to JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries, with later ones taking precedence."""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """Unflatten a dictionary with dot-separated keys."""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def safe_get(data: Dict[str, Any], key_path: str, default: Any = None, sep: str = '.') -> Any:
    """Safely get a value from nested dict using dot notation."""
    keys = key_path.split(sep)
    current = data
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def safe_set(data: Dict[str, Any], key_path: str, value: Any, sep: str = '.') -> None:
    """Safely set a value in nested dict using dot notation."""
    keys = key_path.split(sep)
    current = data
    
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def retry_async(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for async functions with retry logic."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise last_exception
            
        return wrapper
    return decorator


def measure_time():
    """Context manager to measure execution time."""
    class TimeContext:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.duration = None
        
        def __enter__(self):
            self.start_time = time.perf_counter()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.perf_counter()
            self.duration = self.end_time - self.start_time
    
    return TimeContext()


def validate_url(url: str) -> bool:
    """Validate if a string is a valid URL."""
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None


def sanitize_filename(filename: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    import re
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip('. ')
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    return sanitized or 'unnamed'


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


# Legacy compatibility - these functions were in the old utils
def load_docs(*args, **kwargs):
    """Legacy function - use document loaders from components instead."""
    raise NotImplementedError("Use document loaders from llmblocks.components.rag instead")


def load_llm(*args, **kwargs):
    """Legacy function - use LLM providers instead."""
    raise NotImplementedError("Use LLM providers from llmblocks.blocks.llm_provider instead")
