"""
File-based backend for LLMBlocks memory.

This backend stores data in JSON files and provides persistence
across application restarts.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, UTC

from ..base import MemoryBackend, MemoryMessage, MemoryBackendError
from ....utils.logging import get_logger


class FileBackend(MemoryBackend):
    """
    File-based storage backend for memory systems.
    
    Features:
    - Persistent storage in JSON files
    - One file per session
    - Atomic writes for data safety
    - Configurable storage directory
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = get_logger("FileBackend")
        
        # Configuration
        self.storage_dir = Path(config.get("storage_dir", "./memory_data"))
        self.file_extension = config.get("file_extension", ".json")
        self.pretty_print = config.get("pretty_print", True)
        self.backup_files = config.get("backup_files", True)
        
        # Thread safety
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the file backend."""
        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"File backend initialized with storage dir: {self.storage_dir}")
    
    async def close(self) -> None:
        """Close the backend."""
        self.logger.info("File backend closed")
    
    def _get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session."""
        safe_session_id = "".join(c for c in session_id if c.isalnum() or c in "._-")
        return self.storage_dir / f"{safe_session_id}{self.file_extension}"
    
    async def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for a session."""
        async with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = asyncio.Lock()
            return self._locks[session_id]
    
    async def _load_session_data(self, session_id: str) -> List[Dict[str, Any]]:
        """Load session data from file."""
        file_path = self._get_session_file(session_id)
        
        if not file_path.exists():
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("messages", [])
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Failed to load session {session_id}: {e}")
            return []
    
    async def _save_session_data(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Save session data to file."""
        file_path = self._get_session_file(session_id)
        
        # Create backup if enabled
        if self.backup_files and file_path.exists():
            backup_path = file_path.with_suffix(f".backup{self.file_extension}")
            try:
                backup_path.write_bytes(file_path.read_bytes())
            except IOError as e:
                self.logger.warning(f"Failed to create backup for {session_id}: {e}")
        
        # Prepare data
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "message_count": len(messages),
            "messages": messages
        }
        
        # Write to temporary file first, then rename (atomic operation)
        temp_path = file_path.with_suffix(".tmp")
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                if self.pretty_print:
                    json.dump(session_data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(session_data, f, ensure_ascii=False)
            
            # Atomic rename
            temp_path.replace(file_path)
            
        except IOError as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise MemoryBackendError(f"Failed to save session {session_id}: {e}")
    
    async def store_message(self, session_id: str, message: MemoryMessage) -> None:
        """Store a message in file."""
        lock = await self._get_session_lock(session_id)
        
        async with lock:
            # Load existing messages
            messages_data = await self._load_session_data(session_id)
            
            # Add new message
            messages_data.append(message.to_dict())
            
            # Save back to file
            await self._save_session_data(session_id, messages_data)
            
            self.logger.debug(
                f"Stored message in session {session_id}",
                message_id=message.message_id,
                total_messages=len(messages_data)
            )
    
    async def get_messages(
        self, 
        session_id: str, 
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[MemoryMessage]:
        """Retrieve messages from file."""
        lock = await self._get_session_lock(session_id)
        
        async with lock:
            messages_data = await self._load_session_data(session_id)
            
            # Apply offset and limit
            if offset > 0:
                messages_data = messages_data[offset:]
            
            if limit is not None:
                messages_data = messages_data[:limit]
            
            # Convert to MemoryMessage objects
            messages = []
            for msg_data in messages_data:
                try:
                    messages.append(MemoryMessage.from_dict(msg_data))
                except Exception as e:
                    self.logger.warning(f"Failed to parse message: {e}")
                    continue
            
            self.logger.debug(
                f"Retrieved {len(messages)} messages from session {session_id}",
                offset=offset,
                limit=limit
            )
            
            return messages
    
    async def delete_messages(self, session_id: str, message_ids: List[str]) -> int:
        """Delete specific messages."""
        lock = await self._get_session_lock(session_id)
        
        async with lock:
            messages_data = await self._load_session_data(session_id)
            original_count = len(messages_data)
            
            # Filter out messages with matching IDs
            messages_data = [
                msg for msg in messages_data 
                if msg.get("message_id") not in message_ids
            ]
            
            deleted_count = original_count - len(messages_data)
            
            if deleted_count > 0:
                await self._save_session_data(session_id, messages_data)
            
            self.logger.debug(
                f"Deleted {deleted_count} messages from session {session_id}",
                requested_ids=len(message_ids)
            )
            
            return deleted_count
    
    async def clear_session(self, session_id: str) -> int:
        """Clear all messages for a session."""
        lock = await self._get_session_lock(session_id)
        
        async with lock:
            file_path = self._get_session_file(session_id)
            
            if not file_path.exists():
                return 0
            
            # Count messages before deletion
            messages_data = await self._load_session_data(session_id)
            message_count = len(messages_data)
            
            # Remove file
            try:
                file_path.unlink()
                self.logger.debug(f"Cleared session {session_id}", messages_deleted=message_count)
            except IOError as e:
                raise MemoryBackendError(f"Failed to clear session {session_id}: {e}")
            
            return message_count
    
    async def get_sessions(self) -> List[str]:
        """Get list of all session IDs."""
        sessions = []
        
        try:
            for file_path in self.storage_dir.glob(f"*{self.file_extension}"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    # Extract session ID from filename
                    session_id = file_path.stem
                    sessions.append(session_id)
        except OSError as e:
            self.logger.error(f"Failed to list sessions: {e}")
        
        self.logger.debug(f"Found {len(sessions)} sessions")
        return sessions
    
    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        file_path = self._get_session_file(session_id)
        exists = file_path.exists()
        self.logger.debug(f"Session {session_id} exists: {exists}")
        return exists
    
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        file_path = self._get_session_file(session_id)
        
        if not file_path.exists():
            return {
                "exists": False,
                "message_count": 0,
                "file_size_bytes": 0
            }
        
        lock = await self._get_session_lock(session_id)
        
        async with lock:
            try:
                # Get file stats
                file_stat = file_path.stat()
                file_size = file_stat.st_size
                modified_time = datetime.fromtimestamp(file_stat.st_mtime, UTC)
                
                # Load and analyze messages
                messages_data = await self._load_session_data(session_id)
                
                # Role distribution
                role_counts = {}
                for msg_data in messages_data:
                    role = msg_data.get("role", "unknown")
                    role_counts[role] = role_counts.get(role, 0) + 1
                
                # Time range
                if messages_data:
                    timestamps = [
                        datetime.fromisoformat(msg.get("timestamp", "1970-01-01T00:00:00+00:00"))
                        for msg in messages_data
                        if msg.get("timestamp")
                    ]
                    if timestamps:
                        time_span = (max(timestamps) - min(timestamps)).total_seconds()
                    else:
                        time_span = 0
                else:
                    time_span = 0
                
                stats = {
                    "exists": True,
                    "message_count": len(messages_data),
                    "file_size_bytes": file_size,
                    "file_modified": modified_time.isoformat(),
                    "role_distribution": role_counts,
                    "time_span_seconds": time_span,
                    "backend_type": "file"
                }
                
                self.logger.debug(f"Generated stats for session {session_id}", **stats)
                return stats
                
            except Exception as e:
                self.logger.error(f"Failed to get stats for session {session_id}: {e}")
                return {
                    "exists": True,
                    "message_count": 0,
                    "file_size_bytes": 0,
                    "error": str(e)
                }
