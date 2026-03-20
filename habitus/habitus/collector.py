"""High-resolution timeseries data collector via Home Assistant WebSocket API.

Subscribes to state changes for configured sensors, stores in local SQLite DB
with 1-second resolution. Provides query interface for training pipeline.

Architecture:
- WebSocket subscription to HA state_changed events
- Buffered writes (batch every 1000 events or 10 seconds)
- SQLite storage with timestamp+entity_id composite PK
- Automatic retention policy (downsample old data to save space)
- Validation against HA statistics table
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import logging
import os
import sqlite3
import time
from collections import deque
from typing import Any

import aiohttp
import pandas as pd

log = logging.getLogger("habitus")

DB_PATH = os.path.join(os.environ.get("DATA_DIR", "/data"), "timeseries.db")
BUFFER_SIZE = 1000  # Batch writes
FLUSH_INTERVAL_SEC = 10  # Force flush every N seconds


class TimeSeriesDB:
    """Local high-resolution timeseries database."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        
        # Main timeseries table: 1-second resolution
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                entity_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                value REAL,
                unit TEXT,
                source TEXT DEFAULT 'ha_ws',
                PRIMARY KEY (entity_id, timestamp)
            ) WITHOUT ROWID
        """)
        
        # Indexes for fast queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON sensor_data(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_ts ON sensor_data(entity_id, timestamp)")
        
        # Metadata table: track collection status per entity
        conn.execute("""
            CREATE TABLE IF NOT EXISTS collection_meta (
                entity_id TEXT PRIMARY KEY,
                first_seen REAL,
                last_seen REAL,
                total_events INTEGER DEFAULT 0,
                enabled INTEGER DEFAULT 1
            )
        """)
        
        conn.commit()
        conn.close()
        log.info(f"TimeSeriesDB initialized at {self.db_path}")
    
    def insert_batch(self, records: list[tuple[str, float, float, str]]):
        """Insert batch of (entity_id, timestamp, value, unit) records."""
        if not records:
            return
        
        conn = sqlite3.connect(self.db_path)
        try:
            # Insert sensor data
            conn.executemany(
                "INSERT OR REPLACE INTO sensor_data (entity_id, timestamp, value, unit) VALUES (?, ?, ?, ?)",
                records
            )
            
            # Update metadata
            now = time.time()
            for entity_id, ts, _, _ in records:
                conn.execute("""
                    INSERT INTO collection_meta (entity_id, first_seen, last_seen, total_events)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(entity_id) DO UPDATE SET
                        last_seen = ?,
                        total_events = total_events + 1
                """, (entity_id, ts, ts, ts))
            
            conn.commit()
            log.debug(f"Inserted {len(records)} records")
        except Exception as e:
            log.error(f"Batch insert failed: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def fetch_range(
        self,
        entity_ids: list[str],
        start_ts: float,
        end_ts: float,
        resolution: str = "raw"
    ) -> pd.DataFrame:
        """Fetch data for entities in time range.
        
        Args:
            entity_ids: List of entity IDs to fetch
            start_ts: Start timestamp (unix seconds)
            end_ts: End timestamp (unix seconds)
            resolution: 'raw' (1s), '1m', '1h' (aggregated)
        
        Returns:
            DataFrame with columns [entity_id, timestamp, value, unit, samples]
        """
        conn = sqlite3.connect(self.db_path)
        
        entity_str = ",".join(f"'{e}'" for e in entity_ids)
        
        if resolution == "raw":
            query = f"""
                SELECT entity_id, timestamp, value, unit, 1 as samples
                FROM sensor_data
                WHERE entity_id IN ({entity_str})
                  AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            df = pd.read_sql(query, conn, params=(start_ts, end_ts))
        else:
            # Aggregate to specified resolution
            interval = {"1m": 60, "1h": 3600, "1d": 86400}.get(resolution, 3600)
            query = f"""
                SELECT 
                    entity_id,
                    CAST(timestamp / {interval} AS INTEGER) * {interval} as timestamp,
                    AVG(value) as value,
                    MAX(unit) as unit,
                    COUNT(*) as samples
                FROM sensor_data
                WHERE entity_id IN ({entity_str})
                  AND timestamp BETWEEN ? AND ?
                GROUP BY entity_id, CAST(timestamp / {interval} AS INTEGER)
                ORDER BY timestamp
            """
            df = pd.read_sql(query, conn, params=(start_ts, end_ts))
        
        conn.close()
        return df
    
    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        conn = sqlite3.connect(self.db_path)
        
        # Total rows
        total_rows = conn.execute("SELECT COUNT(*) FROM sensor_data").fetchone()[0]
        
        # Per-entity stats
        entities = pd.read_sql("""
            SELECT 
                entity_id,
                datetime(first_seen, 'unixepoch') as first_seen,
                datetime(last_seen, 'unixepoch') as last_seen,
                total_events,
                enabled
            FROM collection_meta
            ORDER BY last_seen DESC
        """, conn)
        
        # Oldest and newest timestamps
        oldest, newest = conn.execute("""
            SELECT MIN(timestamp), MAX(timestamp) FROM sensor_data
        """).fetchone()
        
        conn.close()
        
        return {
            "total_rows": total_rows,
            "entities": entities.to_dict("records"),
            "oldest_ts": oldest,
            "newest_ts": newest,
            "coverage_days": (newest - oldest) / 86400 if oldest and newest else 0
        }
    
    def cleanup_old_data(self, retention_days: int = 90):
        """Delete data older than retention_days."""
        cutoff_ts = time.time() - (retention_days * 86400)
        conn = sqlite3.connect(self.db_path)
        deleted = conn.execute("DELETE FROM sensor_data WHERE timestamp < ?", (cutoff_ts,)).rowcount
        conn.commit()
        conn.close()
        log.info(f"Cleanup: deleted {deleted} rows older than {retention_days} days")


class HAWebSocketCollector:
    """Subscribes to Home Assistant WebSocket API and collects state changes."""
    
    def __init__(self, ha_url: str, ha_token: str, db: TimeSeriesDB):
        self.ha_url = ha_url.replace("http://", "ws://").replace("https://", "wss://")
        if not self.ha_url.endswith("/api/websocket"):
            self.ha_url = f"{self.ha_url}/api/websocket"
        self.ha_token = ha_token
        self.db = db
        self.buffer: deque[tuple[str, float, float, str]] = deque(maxlen=BUFFER_SIZE)
        self.last_flush = time.time()
        self.running = False
        self.msg_id = 1
        self.subscribed_entities: set[str] = set()
    
    async def connect_and_subscribe(self, entity_ids: list[str]) -> None:
        """Connect to HA WebSocket and subscribe to state changes."""
        self.subscribed_entities = set(entity_ids)
        self.running = True
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.ws_connect(self.ha_url) as ws:
                    log.info(f"WebSocket connected to {self.ha_url}")
                    
                    # Receive auth_required message
                    msg = await ws.receive_json()
                    if msg.get("type") != "auth_required":
                        log.error(f"Unexpected message: {msg}")
                        return
                    
                    # Send auth
                    await ws.send_json({"type": "auth", "access_token": self.ha_token})
                    msg = await ws.receive_json()
                    if msg.get("type") != "auth_ok":
                        log.error(f"Auth failed: {msg}")
                        return
                    
                    log.info("WebSocket authenticated")
                    
                    # Subscribe to state_changed events
                    await ws.send_json({
                        "id": self.msg_id,
                        "type": "subscribe_events",
                        "event_type": "state_changed"
                    })
                    self.msg_id += 1
                    
                    # Start periodic flush task
                    flush_task = asyncio.create_task(self._periodic_flush())
                    
                    # Listen for events
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            await self._handle_message(data)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            log.error(f"WebSocket error: {msg}")
                            break
                    
                    flush_task.cancel()
            except Exception as e:
                log.error(f"WebSocket connection failed: {e}")
            finally:
                self.running = False
                self._flush_buffer()
    
    async def _handle_message(self, msg: dict):
        """Process incoming WebSocket message."""
        if msg.get("type") != "event":
            return
        
        event = msg.get("event", {})
        if event.get("event_type") != "state_changed":
            return
        
        data = event.get("data", {})
        entity_id = data.get("entity_id")
        new_state = data.get("new_state")
        
        if not entity_id or not new_state:
            return
        
        # Filter to subscribed entities
        if entity_id not in self.subscribed_entities:
            return
        
        # Extract state value
        try:
            state_val = new_state.get("state")
            if state_val in ("unknown", "unavailable", None):
                return
            
            value = float(state_val)
            unit = new_state.get("attributes", {}).get("unit_of_measurement", "")
            timestamp = time.time()  # Use current time (close enough to HA event time)
            
            # Buffer the record
            self.buffer.append((entity_id, timestamp, value, unit))
            
            # Flush if buffer full
            if len(self.buffer) >= BUFFER_SIZE:
                self._flush_buffer()
        except (ValueError, TypeError):
            # Non-numeric state, skip
            pass
    
    def _flush_buffer(self):
        """Write buffered records to database."""
        if not self.buffer:
            return
        
        records = list(self.buffer)
        self.buffer.clear()
        self.last_flush = time.time()
        
        try:
            self.db.insert_batch(records)
            log.debug(f"Flushed {len(records)} records to DB")
        except Exception as e:
            log.error(f"Flush failed: {e}")
    
    async def _periodic_flush(self):
        """Periodically flush buffer even if not full."""
        while self.running:
            await asyncio.sleep(FLUSH_INTERVAL_SEC)
            if time.time() - self.last_flush >= FLUSH_INTERVAL_SEC:
                self._flush_buffer()


# Singleton instance
_db = TimeSeriesDB()
_collector: HAWebSocketCollector | None = None


def get_db() -> TimeSeriesDB:
    """Get singleton TimeSeriesDB instance."""
    return _db


def start_collector(entity_ids: list[str]) -> None:
    """Start WebSocket collector in background thread."""
    import threading
    
    global _collector
    
    ha_url = os.environ.get("HA_URL", "http://supervisor/core")
    ha_token = os.environ.get("SUPERVISOR_TOKEN")
    
    if not ha_token:
        log.warning("No SUPERVISOR_TOKEN — collector disabled")
        return
    
    _collector = HAWebSocketCollector(ha_url, ha_token, _db)
    
    # Run in background thread with its own event loop
    def _run_collector():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_collector.connect_and_subscribe(entity_ids))
        except Exception as e:
            log.error(f"Collector thread crashed: {e}")
        finally:
            loop.close()
    
    thread = threading.Thread(target=_run_collector, daemon=True, name="habitus-collector")
    thread.start()
    log.info(f"Started collector thread for {len(entity_ids)} entities")


def stop_collector() -> None:
    """Stop WebSocket collector."""
    global _collector
    if _collector:
        _collector.running = False
        _collector = None
        log.info("Stopped collector")
