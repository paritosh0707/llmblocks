"""
Redis backend for LLMBlocks memory.

This backend uses Redis for high-performance, distributed memory storage
with optional persistence and clustering support.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, UTC

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

from ..base import MemoryBackend, MemoryMessage, MemoryBackendError
from ....utils.logging import get_logger


class RedisBackend(MemoryBackend):
    """
    Redis-based storage backend for memory systems.
    
    Features:
    - High-performance in-memory storage
    - Optional persistence to disk
    - Distributed/clustered support
    - Automatic expiration (TTL)
    - Atomic operations
    """
    
    def __init__(self, config: Dict[str, Any]):
        if not REDIS_AVAILABLE:
            raise MemoryBackendError("Redis is not available. Install with: pip install redis")
        
        super().__init__(config)
        self.logger = get_logger("RedisBackend")
        
        # Redis configuration
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6379)
        self.db = config.get("db", 0)
        self.password = config.get("password")
        self.username = config.get("username")
        self.ssl = config.get("ssl", False)
        
        # Connection pool settings
        self.max_connections = config.get("max_connections", 10)
        self.retry_on_timeout = config.get("retry_on_timeout", True)
        self.socket_timeout = config.get("socket_timeout", 5.0)
        self.socket_connect_timeout = config.get("socket_connect_timeout", 5.0)
        
        # Key settings
        self.key_prefix = config.get("key_prefix", "llmblocks:memory:")
        self.session_ttl = config.get("session_ttl", 86400 * 7)  # 7 days default
        
        # Serialization
        self.compression = config.get("compression", False)
        
        # Redis client
        self.redis_client: Optional[redis.Redis] = None
    
    async def initialize(self) -> None:
        """Initialize the Redis backend."""
        try:
            # Create Redis connection
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                username=self.username,
                ssl=self.ssl,
                max_connections=self.max_connections,
                retry_on_timeout=self.retry_on_timeout,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            self.logger.info(
                f"Redis backend initialized",
                host=self.host,
                port=self.port,
                db=self.db
            )
            
        except Exception as e:
            raise MemoryBackendError(f"Failed to initialize Redis backend: {e}")
    
    async def close(self) -> None:
        """Close the Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
        self.logger.info("Redis backend closed")
    
    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for a session."""
        return f"{self.key_prefix}session:{session_id}"
    
    def _get_session_meta_key(self, session_id: str) -> str:
        """Get Redis key for session metadata."""
        return f"{self.key_prefix}meta:{session_id}"
    
    def _get_sessions_set_key(self) -> str:
        """Get Redis key for the set of all sessions."""
        return f"{self.key_prefix}sessions"
    
    async def store_message(self, session_id: str, message: MemoryMessage) -> None:
        """Store a message in Redis."""
        if not self.redis_client:
            raise MemoryBackendError("Redis client not initialized")
        
        try:
            session_key = self._get_session_key(session_id)
            meta_key = self._get_session_meta_key(session_id)
            sessions_key = self._get_sessions_set_key()
            
            # Serialize message
            message_data = json.dumps(message.to_dict())
            
            # Use pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Add message to list
            pipe.lpush(session_key, message_data)
            
            # Update session metadata
            pipe.hset(meta_key, mapping={
                "last_access": datetime.now(UTC).isoformat(),
                "message_count": await self.redis_client.llen(session_key) + 1
            })
            
            # Add session to sessions set
            pipe.sadd(sessions_key, session_id)
            
            # Set TTL if configured
            if self.session_ttl:
                pipe.expire(session_key, self.session_ttl)
                pipe.expire(meta_key, self.session_ttl)
            
            # Execute pipeline
            await pipe.execute()
            
            self.logger.debug(
                f"Stored message in session {session_id}",
                message_id=message.message_id,
                role=message.role.value
            )
            
        except Exception as e:
            raise MemoryBackendError(f"Failed to store message: {e}")
    
    async def get_messages(
        self, 
        session_id: str, 
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[MemoryMessage]:
        """Retrieve messages from Redis."""
        if not self.redis_client:
            raise MemoryBackendError("Redis client not initialized")
        
        try:
            session_key = self._get_session_key(session_id)
            meta_key = self._get_session_meta_key(session_id)
            
            # Update last access time
            await self.redis_client.hset(
                meta_key, 
                "last_access", 
                datetime.now(UTC).isoformat()
            )
            
            # Calculate range for Redis LRANGE (Redis uses 0-based indexing)
            start = offset
            if limit is not None:
                end = offset + limit - 1
            else:
                end = -1  # Get all remaining items
            
            # Get messages (Redis LPUSH stores newest first, so we reverse)
            message_data_list = await self.redis_client.lrange(session_key, start, end)
            
            # Parse messages
            messages = []
            for message_data in reversed(message_data_list):  # Reverse to get chronological order
                try:
                    msg_dict = json.loads(message_data)
                    messages.append(MemoryMessage.from_dict(msg_dict))
                except Exception as e:
                    self.logger.warning(f"Failed to parse message: {e}")
                    continue
            
            self.logger.debug(
                f"Retrieved {len(messages)} messages from session {session_id}",
                offset=offset,
                limit=limit
            )
            
            return messages
            
        except Exception as e:
            raise MemoryBackendError(f"Failed to get messages: {e}")
    
    async def delete_messages(self, session_id: str, message_ids: List[str]) -> int:
        """Delete specific messages."""
        if not self.redis_client:
            raise MemoryBackendError("Redis client not initialized")
        
        try:
            session_key = self._get_session_key(session_id)
            
            # Get all messages
            all_message_data = await self.redis_client.lrange(session_key, 0, -1)
            
            # Filter out messages to delete
            remaining_messages = []
            deleted_count = 0
            
            for message_data in all_message_data:
                try:
                    msg_dict = json.loads(message_data)
                    if msg_dict.get("message_id") not in message_ids:
                        remaining_messages.append(message_data)
                    else:
                        deleted_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to parse message during deletion: {e}")
                    remaining_messages.append(message_data)  # Keep unparseable messages
            
            if deleted_count > 0:
                # Replace the entire list
                pipe = self.redis_client.pipeline()
                pipe.delete(session_key)
                if remaining_messages:
                    pipe.lpush(session_key, *remaining_messages)
                
                # Update metadata
                meta_key = self._get_session_meta_key(session_id)
                pipe.hset(meta_key, mapping={
                    "last_access": datetime.now(UTC).isoformat(),
                    "message_count": len(remaining_messages)
                })
                
                await pipe.execute()
            
            self.logger.debug(
                f"Deleted {deleted_count} messages from session {session_id}",
                requested_ids=len(message_ids)
            )
            
            return deleted_count
            
        except Exception as e:
            raise MemoryBackendError(f"Failed to delete messages: {e}")
    
    async def clear_session(self, session_id: str) -> int:
        """Clear all messages for a session."""
        if not self.redis_client:
            raise MemoryBackendError("Redis client not initialized")
        
        try:
            session_key = self._get_session_key(session_id)
            meta_key = self._get_session_meta_key(session_id)
            sessions_key = self._get_sessions_set_key()
            
            # Get message count before deletion
            message_count = await self.redis_client.llen(session_key)
            
            # Delete session data
            pipe = self.redis_client.pipeline()
            pipe.delete(session_key)
            pipe.delete(meta_key)
            pipe.srem(sessions_key, session_id)
            await pipe.execute()
            
            self.logger.debug(f"Cleared session {session_id}", messages_deleted=message_count)
            return message_count
            
        except Exception as e:
            raise MemoryBackendError(f"Failed to clear session: {e}")
    
    async def get_sessions(self) -> List[str]:
        """Get list of all session IDs."""
        if not self.redis_client:
            raise MemoryBackendError("Redis client not initialized")
        
        try:
            sessions_key = self._get_sessions_set_key()
            sessions = await self.redis_client.smembers(sessions_key)
            
            # Convert to list and sort
            session_list = sorted(list(sessions))
            
            self.logger.debug(f"Retrieved {len(session_list)} session IDs")
            return session_list
            
        except Exception as e:
            raise MemoryBackendError(f"Failed to get sessions: {e}")
    
    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        if not self.redis_client:
            raise MemoryBackendError("Redis client not initialized")
        
        try:
            session_key = self._get_session_key(session_id)
            exists = await self.redis_client.exists(session_key) > 0
            
            self.logger.debug(f"Session {session_id} exists: {exists}")
            return exists
            
        except Exception as e:
            raise MemoryBackendError(f"Failed to check session existence: {e}")
    
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        if not self.redis_client:
            raise MemoryBackendError("Redis client not initialized")
        
        try:
            session_key = self._get_session_key(session_id)
            meta_key = self._get_session_meta_key(session_id)
            
            # Check if session exists
            if not await self.redis_client.exists(session_key):
                return {
                    "exists": False,
                    "message_count": 0,
                    "memory_usage_bytes": 0
                }
            
            # Get basic stats
            message_count = await self.redis_client.llen(session_key)
            memory_usage = await self.redis_client.memory_usage(session_key) or 0
            ttl = await self.redis_client.ttl(session_key)
            
            # Get metadata
            metadata = await self.redis_client.hgetall(meta_key)
            
            # Analyze message roles (sample recent messages for performance)
            sample_messages = await self.redis_client.lrange(session_key, 0, 99)  # Last 100 messages
            role_counts = {}
            
            for message_data in sample_messages:
                try:
                    msg_dict = json.loads(message_data)
                    role = msg_dict.get("role", "unknown")
                    role_counts[role] = role_counts.get(role, 0) + 1
                except Exception:
                    continue
            
            stats = {
                "exists": True,
                "message_count": message_count,
                "memory_usage_bytes": memory_usage,
                "ttl_seconds": ttl if ttl > 0 else None,
                "last_access": metadata.get("last_access"),
                "role_distribution": role_counts,
                "backend_type": "redis"
            }
            
            self.logger.debug(f"Generated stats for session {session_id}", **stats)
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get stats for session {session_id}: {e}")
            return {
                "exists": False,
                "message_count": 0,
                "memory_usage_bytes": 0,
                "error": str(e)
            }
    
    async def search_messages(
        self, 
        session_id: str, 
        query: str, 
        limit: int = 10
    ) -> List[MemoryMessage]:
        """Search messages by content."""
        if not self.redis_client:
            raise MemoryBackendError("Redis client not initialized")
        
        try:
            # For basic search, we need to get all messages and filter
            # In a production system, you might use Redis Search module
            all_messages = await self.get_messages(session_id)
            
            query_lower = query.lower()
            results = []
            
            for message in all_messages:
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
            
        except Exception as e:
            raise MemoryBackendError(f"Failed to search messages: {e}")
    
    # Redis-specific utility methods
    async def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        if not self.redis_client:
            raise MemoryBackendError("Redis client not initialized")
        
        try:
            info = await self.redis_client.info()
            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses")
            }
        except Exception as e:
            raise MemoryBackendError(f"Failed to get Redis info: {e}")
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions (Redis handles this automatically with TTL)."""
        # Redis automatically handles TTL cleanup, but we can clean up the sessions set
        if not self.redis_client:
            raise MemoryBackendError("Redis client not initialized")
        
        try:
            sessions_key = self._get_sessions_set_key()
            all_sessions = await self.redis_client.smembers(sessions_key)
            
            expired_sessions = []
            for session_id in all_sessions:
                session_key = self._get_session_key(session_id)
                if not await self.redis_client.exists(session_key):
                    expired_sessions.append(session_id)
            
            if expired_sessions:
                await self.redis_client.srem(sessions_key, *expired_sessions)
            
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            return len(expired_sessions)
            
        except Exception as e:
            raise MemoryBackendError(f"Failed to cleanup expired sessions: {e}")
