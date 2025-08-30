"""
In-memory backend for LLMBlocks memory.

This backend stores all data in memory and is suitable for
development, testing, and applications that don't need persistence.
"""

import asyncio
from typing import Any, Dict, List, Optional
from collections import defaultdict
from datetime import datetime, UTC

from ..base import MemoryBackend, MemoryMessage, MemoryBackendError
from ....utils.logging import get_logger


class InMemoryBackend(MemoryBackend):
    """
    In-memory storage backend for memory systems.
    
    Features:
    - Fast access (all data in RAM)
    - No persistence (data lost on restart)
    - Thread-safe operations
    - Suitable for development and testing
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = get_logger("InMemoryBackend")
        
        # Storage
        self._sessions: Dict[str, List[MemoryMessage]] = defaultdict(list)
        self._session_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Configuration
        self.max_sessions = config.get("max_sessions", 1000)
        self.max_messages_per_session = config.get("max_messages_per_session", 10000)
        
        # Thread safety
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the in-memory backend."""
        self.logger.info("In-memory backend initialized")
    
    async def close(self) -> None:
        """Close the backend and cleanup."""
        async with self._lock:
            self._sessions.clear()
            self._session_metadata.clear()
        self.logger.info("In-memory backend closed")
    
    async def store_message(self, session_id: str, message: MemoryMessage) -> None:
        """Store a message in memory."""
        async with self._lock:
            # Check session limits
            if len(self._sessions) >= self.max_sessions and session_id not in self._sessions:
                # Remove oldest session
                oldest_session = min(
                    self._sessions.keys(),
                    key=lambda s: self._session_metadata[s].get("last_access", datetime.min)
                )
                del self._sessions[oldest_session]
                del self._session_metadata[oldest_session]
                self.logger.debug(f"Removed oldest session: {oldest_session}")
            
            # Check message limits per session
            session_messages = self._sessions[session_id]
            if len(session_messages) >= self.max_messages_per_session:
                # Remove oldest messages (keep half)
                keep_count = self.max_messages_per_session // 2
                self._sessions[session_id] = session_messages[-keep_count:]
                self.logger.debug(f"Trimmed session {session_id} to {keep_count} messages")
            
            # Store message
            self._sessions[session_id].append(message)
            
            # Update metadata
            self._session_metadata[session_id].update({
                "last_access": datetime.now(UTC),
                "message_count": len(self._sessions[session_id])
            })
            
            self.logger.debug(
                f"Stored message in session {session_id}",
                message_id=message.message_id,
                role=message.role.value
            )
    
    async def get_messages(
        self, 
        session_id: str, 
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[MemoryMessage]:
        """Retrieve messages from memory."""
        async with self._lock:
            if session_id not in self._sessions:
                return []
            
            messages = self._sessions[session_id]
            
            # Update last access
            self._session_metadata[session_id]["last_access"] = datetime.now(UTC)
            
            # Apply offset and limit
            if offset > 0:
                messages = messages[offset:]
            
            if limit is not None:
                messages = messages[:limit]
            
            self.logger.debug(
                f"Retrieved {len(messages)} messages from session {session_id}",
                total_messages=len(self._sessions[session_id]),
                offset=offset,
                limit=limit
            )
            
            return messages.copy()  # Return a copy to prevent external modification
    
    async def delete_messages(self, session_id: str, message_ids: List[str]) -> int:
        """Delete specific messages."""
        async with self._lock:
            if session_id not in self._sessions:
                return 0
            
            messages = self._sessions[session_id]
            original_count = len(messages)
            
            # Filter out messages with matching IDs
            self._sessions[session_id] = [
                msg for msg in messages 
                if msg.message_id not in message_ids
            ]
            
            deleted_count = original_count - len(self._sessions[session_id])
            
            # Update metadata
            self._session_metadata[session_id].update({
                "last_access": datetime.now(UTC),
                "message_count": len(self._sessions[session_id])
            })
            
            self.logger.debug(
                f"Deleted {deleted_count} messages from session {session_id}",
                requested_ids=len(message_ids)
            )
            
            return deleted_count
    
    async def clear_session(self, session_id: str) -> int:
        """Clear all messages for a session."""
        async with self._lock:
            if session_id not in self._sessions:
                return 0
            
            message_count = len(self._sessions[session_id])
            del self._sessions[session_id]
            del self._session_metadata[session_id]
            
            self.logger.debug(f"Cleared session {session_id}", messages_deleted=message_count)
            return message_count
    
    async def get_sessions(self) -> List[str]:
        """Get list of all session IDs."""
        async with self._lock:
            sessions = list(self._sessions.keys())
            self.logger.debug(f"Retrieved {len(sessions)} session IDs")
            return sessions
    
    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        async with self._lock:
            exists = session_id in self._sessions
            self.logger.debug(f"Session {session_id} exists: {exists}")
            return exists
    
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        async with self._lock:
            if session_id not in self._sessions:
                return {
                    "exists": False,
                    "message_count": 0,
                    "total_size_bytes": 0
                }
            
            messages = self._sessions[session_id]
            metadata = self._session_metadata[session_id]
            
            # Calculate approximate size
            total_size = sum(
                len(msg.content.encode('utf-8')) + 
                len(str(msg.metadata).encode('utf-8'))
                for msg in messages
            )
            
            # Role distribution
            role_counts = {}
            for msg in messages:
                role = msg.role.value
                role_counts[role] = role_counts.get(role, 0) + 1
            
            # Time range
            if messages:
                first_message = min(messages, key=lambda m: m.timestamp)
                last_message = max(messages, key=lambda m: m.timestamp)
                time_span = (last_message.timestamp - first_message.timestamp).total_seconds()
            else:
                time_span = 0
            
            stats = {
                "exists": True,
                "message_count": len(messages),
                "total_size_bytes": total_size,
                "role_distribution": role_counts,
                "time_span_seconds": time_span,
                "last_access": metadata.get("last_access", datetime.now(UTC)).isoformat(),
                "backend_type": "in_memory"
            }
            
            self.logger.debug(f"Generated stats for session {session_id}", **stats)
            return stats
    
    async def search_messages(
        self, 
        session_id: str, 
        query: str, 
        limit: int = 10
    ) -> List[MemoryMessage]:
        """Search messages by content."""
        async with self._lock:
            if session_id not in self._sessions:
                return []
            
            messages = self._sessions[session_id]
            query_lower = query.lower()
            results = []
            
            for message in messages:
                if query_lower in message.content.lower():
                    results.append(message)
                    if len(results) >= limit:
                        break
            
            self.logger.debug(
                f"Search in session {session_id} found {len(results)} results",
                query=query,
                limit=limit
            )
            
            return results
    
    async def get_message_by_id(self, session_id: str, message_id: str) -> Optional[MemoryMessage]:
        """Get a specific message by ID."""
        async with self._lock:
            if session_id not in self._sessions:
                return None
            
            for message in self._sessions[session_id]:
                if message.message_id == message_id:
                    self.logger.debug(f"Found message {message_id} in session {session_id}")
                    return message
            
            self.logger.debug(f"Message {message_id} not found in session {session_id}")
            return None
    
    # Additional utility methods
    async def get_backend_stats(self) -> Dict[str, Any]:
        """Get overall backend statistics."""
        async with self._lock:
            total_messages = sum(len(messages) for messages in self._sessions.values())
            total_size = 0
            
            for messages in self._sessions.values():
                for msg in messages:
                    total_size += len(msg.content.encode('utf-8'))
            
            return {
                "backend_type": "in_memory",
                "total_sessions": len(self._sessions),
                "total_messages": total_messages,
                "total_size_bytes": total_size,
                "max_sessions": self.max_sessions,
                "max_messages_per_session": self.max_messages_per_session,
                "memory_usage_mb": total_size / (1024 * 1024)
            }
    
    async def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old sessions that haven't been accessed recently."""
        async with self._lock:
            cutoff_time = datetime.now(UTC).timestamp() - (max_age_hours * 3600)
            sessions_to_remove = []
            
            for session_id, metadata in self._session_metadata.items():
                last_access = metadata.get("last_access", datetime.min)
                if isinstance(last_access, str):
                    last_access = datetime.fromisoformat(last_access)
                
                if last_access.timestamp() < cutoff_time:
                    sessions_to_remove.append(session_id)
            
            # Remove old sessions
            for session_id in sessions_to_remove:
                del self._sessions[session_id]
                del self._session_metadata[session_id]
            
            self.logger.info(
                f"Cleaned up {len(sessions_to_remove)} old sessions",
                max_age_hours=max_age_hours
            )
            
            return len(sessions_to_remove)
