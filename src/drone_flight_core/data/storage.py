"""
Data storage abstraction for drone flight data.

Provides unified interface for storing different types of data
with support for local file storage and future cloud backends.
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterator
import threading
import structlog

logger = structlog.get_logger(__name__)


class StorageType(Enum):
    """Type of storage backend."""

    LOCAL_FILE = auto()
    SQLITE = auto()
    # Future: S3, AZURE_BLOB, etc.


@dataclass
class StorageMetrics:
    """Storage usage metrics."""

    total_bytes: int = 0
    file_count: int = 0
    oldest_record: datetime | None = None
    newest_record: datetime | None = None


class StorageBackend(ABC):
    """Abstract storage backend interface."""

    @abstractmethod
    def store(self, key: str, data: Any, metadata: dict | None = None) -> bool:
        """Store data with a key."""
        pass

    @abstractmethod
    def retrieve(self, key: str) -> Any | None:
        """Retrieve data by key."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data by key."""
        pass

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with optional prefix filter."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def get_metrics(self) -> StorageMetrics:
        """Get storage metrics."""
        pass


class LocalFileStorage(StorageBackend):
    """Local file system storage backend."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self._base_dir / ".metadata.json"
        self._metadata: dict[str, dict] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata index."""
        if self._metadata_file.exists():
            with open(self._metadata_file) as f:
                self._metadata = json.load(f)

    def _save_metadata(self) -> None:
        """Save metadata index."""
        with open(self._metadata_file, "w") as f:
            json.dump(self._metadata, f)

    def _key_to_path(self, key: str) -> Path:
        """Convert key to file path."""
        # Replace path separators to create directory structure
        parts = key.split("/")
        return self._base_dir.joinpath(*parts)

    def store(self, key: str, data: Any, metadata: dict | None = None) -> bool:
        """Store data to local file."""
        try:
            file_path = self._key_to_path(key)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine format based on data type
            if isinstance(data, (dict, list)):
                with open(file_path.with_suffix(".json"), "w") as f:
                    json.dump(data, f)
            elif isinstance(data, bytes):
                with open(file_path, "wb") as f:
                    f.write(data)
            elif isinstance(data, str):
                with open(file_path.with_suffix(".txt"), "w") as f:
                    f.write(data)
            else:
                # Serialize as JSON
                with open(file_path.with_suffix(".json"), "w") as f:
                    json.dump({"data": str(data)}, f)

            # Update metadata
            self._metadata[key] = {
                "stored_at": datetime.now().isoformat(),
                "size": file_path.stat().st_size if file_path.exists() else 0,
                **(metadata or {}),
            }
            self._save_metadata()

            return True

        except Exception as e:
            logger.error("Storage error", key=key, error=str(e))
            return False

    def retrieve(self, key: str) -> Any | None:
        """Retrieve data from local file."""
        file_path = self._key_to_path(key)

        # Try different extensions
        for suffix in [".json", ".txt", ""]:
            path = file_path.with_suffix(suffix) if suffix else file_path
            if path.exists():
                try:
                    if suffix == ".json":
                        with open(path) as f:
                            return json.load(f)
                    elif suffix == ".txt":
                        with open(path) as f:
                            return f.read()
                    else:
                        with open(path, "rb") as f:
                            return f.read()
                except Exception as e:
                    logger.error("Retrieve error", key=key, error=str(e))

        return None

    def delete(self, key: str) -> bool:
        """Delete local file."""
        file_path = self._key_to_path(key)

        deleted = False
        for suffix in [".json", ".txt", ""]:
            path = file_path.with_suffix(suffix) if suffix else file_path
            if path.exists():
                path.unlink()
                deleted = True

        if key in self._metadata:
            del self._metadata[key]
            self._save_metadata()

        return deleted

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with prefix."""
        keys = []
        prefix_path = self._base_dir / prefix if prefix else self._base_dir

        if prefix_path.exists():
            for path in prefix_path.rglob("*"):
                if path.is_file() and not path.name.startswith("."):
                    # Convert path back to key
                    rel_path = path.relative_to(self._base_dir)
                    key = str(rel_path).replace("\\", "/")
                    # Remove extension
                    if key.endswith((".json", ".txt")):
                        key = key.rsplit(".", 1)[0]
                    keys.append(key)

        return keys

    def exists(self, key: str) -> bool:
        """Check if file exists."""
        file_path = self._key_to_path(key)
        return any(
            file_path.with_suffix(s).exists() if s else file_path.exists()
            for s in [".json", ".txt", ""]
        )

    def get_metrics(self) -> StorageMetrics:
        """Get storage metrics."""
        total_bytes = 0
        file_count = 0
        oldest = None
        newest = None

        for path in self._base_dir.rglob("*"):
            if path.is_file() and not path.name.startswith("."):
                file_count += 1
                total_bytes += path.stat().st_size
                mtime = datetime.fromtimestamp(path.stat().st_mtime)

                if oldest is None or mtime < oldest:
                    oldest = mtime
                if newest is None or mtime > newest:
                    newest = mtime

        return StorageMetrics(
            total_bytes=total_bytes,
            file_count=file_count,
            oldest_record=oldest,
            newest_record=newest,
        )


class SQLiteStorage(StorageBackend):
    """SQLite database storage backend."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS storage (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_key_prefix ON storage (key)
            """)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def store(self, key: str, data: Any, metadata: dict | None = None) -> bool:
        """Store data in SQLite."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    # Determine data type
                    if isinstance(data, (dict, list)):
                        data_str = json.dumps(data)
                        data_type = "json"
                    elif isinstance(data, bytes):
                        import base64
                        data_str = base64.b64encode(data).decode()
                        data_type = "bytes"
                    else:
                        data_str = str(data)
                        data_type = "string"

                    metadata_str = json.dumps(metadata) if metadata else None

                    conn.execute("""
                        INSERT OR REPLACE INTO storage (key, data, data_type, metadata, updated_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (key, data_str, data_type, metadata_str))
                    conn.commit()

            return True

        except Exception as e:
            logger.error("SQLite store error", key=key, error=str(e))
            return False

    def retrieve(self, key: str) -> Any | None:
        """Retrieve data from SQLite."""
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT data, data_type FROM storage WHERE key = ?",
                    (key,)
                ).fetchone()

                if row:
                    data_str = row["data"]
                    data_type = row["data_type"]

                    if data_type == "json":
                        return json.loads(data_str)
                    elif data_type == "bytes":
                        import base64
                        return base64.b64decode(data_str)
                    else:
                        return data_str

            return None

        except Exception as e:
            logger.error("SQLite retrieve error", key=key, error=str(e))
            return None

    def delete(self, key: str) -> bool:
        """Delete from SQLite."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.execute("DELETE FROM storage WHERE key = ?", (key,))
                    conn.commit()
                    return cursor.rowcount > 0

        except Exception as e:
            logger.error("SQLite delete error", key=key, error=str(e))
            return False

    def list_keys(self, prefix: str = "") -> list[str]:
        """List keys with prefix."""
        try:
            with self._get_connection() as conn:
                if prefix:
                    rows = conn.execute(
                        "SELECT key FROM storage WHERE key LIKE ?",
                        (f"{prefix}%",)
                    ).fetchall()
                else:
                    rows = conn.execute("SELECT key FROM storage").fetchall()

                return [row["key"] for row in rows]

        except Exception as e:
            logger.error("SQLite list error", error=str(e))
            return []

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT 1 FROM storage WHERE key = ?",
                    (key,)
                ).fetchone()
                return row is not None

        except Exception as e:
            logger.error("SQLite exists error", key=key, error=str(e))
            return False

    def get_metrics(self) -> StorageMetrics:
        """Get storage metrics."""
        try:
            with self._get_connection() as conn:
                row = conn.execute("""
                    SELECT
                        COUNT(*) as count,
                        MIN(created_at) as oldest,
                        MAX(updated_at) as newest
                    FROM storage
                """).fetchone()

                return StorageMetrics(
                    total_bytes=self._db_path.stat().st_size if self._db_path.exists() else 0,
                    file_count=row["count"],
                    oldest_record=datetime.fromisoformat(row["oldest"]) if row["oldest"] else None,
                    newest_record=datetime.fromisoformat(row["newest"]) if row["newest"] else None,
                )

        except Exception as e:
            logger.error("SQLite metrics error", error=str(e))
            return StorageMetrics()


class DataStorage:
    """
    Unified data storage interface.

    Provides consistent API regardless of backend storage type.
    """

    def __init__(
        self,
        storage_type: StorageType,
        base_path: Path,
    ) -> None:
        self._storage_type = storage_type
        self._base_path = Path(base_path)

        if storage_type == StorageType.LOCAL_FILE:
            self._backend = LocalFileStorage(self._base_path)
        elif storage_type == StorageType.SQLITE:
            self._backend = SQLiteStorage(self._base_path / "data.db")
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

        logger.info(
            "Data storage initialized",
            storage_type=storage_type.name,
            path=str(base_path),
        )

    def store(self, key: str, data: Any, metadata: dict | None = None) -> bool:
        """Store data with a key."""
        return self._backend.store(key, data, metadata)

    def retrieve(self, key: str) -> Any | None:
        """Retrieve data by key."""
        return self._backend.retrieve(key)

    def delete(self, key: str) -> bool:
        """Delete data by key."""
        return self._backend.delete(key)

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with optional prefix."""
        return self._backend.list_keys(prefix)

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self._backend.exists(key)

    def get_metrics(self) -> StorageMetrics:
        """Get storage metrics."""
        return self._backend.get_metrics()

    # Convenience methods for common data types

    def store_config(self, name: str, config: dict) -> bool:
        """Store configuration."""
        return self.store(f"config/{name}", config)

    def load_config(self, name: str) -> dict | None:
        """Load configuration."""
        return self.retrieve(f"config/{name}")

    def store_waypoints(self, mission_id: str, waypoints: list[dict]) -> bool:
        """Store mission waypoints."""
        return self.store(f"missions/{mission_id}/waypoints", waypoints)

    def load_waypoints(self, mission_id: str) -> list[dict] | None:
        """Load mission waypoints."""
        return self.retrieve(f"missions/{mission_id}/waypoints")

    def to_dict(self) -> dict[str, Any]:
        """Export storage status."""
        metrics = self.get_metrics()
        return {
            "storage_type": self._storage_type.name,
            "base_path": str(self._base_path),
            "total_bytes": metrics.total_bytes,
            "file_count": metrics.file_count,
        }
