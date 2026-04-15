"""WebSocket service for monitoring ESP32 thermal data streams."""

import asyncio
import base64
import io
import json
import logging
import threading
from typing import Optional, Callable

import numpy as np
import websockets


logger = logging.getLogger(__name__)

THERMAL_WIDTH = 32
THERMAL_HEIGHT = 24


class WebSocketService:
    """
    Background WebSocket client for ESP32 thermal data monitoring.
    
    Runs in a separate thread to continuously listen for thermal frames
    and invoke registered callbacks.
    """
    
    def __init__(self, uri: str, on_frame_callback: Optional[Callable] = None):
        """
        Initialize WebSocket service.
        
        Args:
            uri: WebSocket URI of ESP32 server.
            on_frame_callback: Optional callback function for each frame.
        """
        self.uri = uri
        self.on_frame_callback = on_frame_callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def start(self) -> None:
        """
        Start the WebSocket monitoring in background thread.
        
        Raises:
            RuntimeError: If service is already running.
        """
        if self._running:
            raise RuntimeError("WebSocket service already running")
        
        self._running = True
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        logger.info(f"WebSocket service started (URI: {self.uri})")
    
    def stop(self) -> None:
        """Stop the WebSocket monitoring."""
        if not self._running:
            return
        
        self._running = False
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._cleanup(), self._loop)
        
        if self._thread:
            self._thread.join(timeout=5)
        
        logger.info("WebSocket service stopped")
    
    def set_frame_callback(self, callback: Callable) -> None:
        """
        Register callback for incoming frames.
        
        Args:
            callback: Function to call with thermal frame data.
        """
        self.on_frame_callback = callback
    
    def _run_event_loop(self) -> None:
        """Run asyncio event loop in thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._connect_and_listen())
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            self._loop.close()
    
    async def _connect_and_listen(self) -> None:
        """
        Connect to ESP32 and listen for thermal frames.
        
        Implements reconnection logic with exponential backoff.
        """
        reconnect_delay = 2
        max_delay = 30
        
        while self._running:
            try:
                logger.info(f"Connecting to {self.uri}")
                async with websockets.connect(self.uri, max_size=None) as ws:
                    logger.info("Connected to ESP32")
                    reconnect_delay = 2  # Reset delay on successful connection
                    
                    async for message in ws:
                        if not self._running:
                            break
                        
                        try:
                            frame = self._decode_payload(message)
                            if frame is not None and self.on_frame_callback:
                                self.on_frame_callback(frame)
                        except Exception as decode_error:
                            logger.warning(f"Frame decode error: {decode_error}")
            
            except websockets.exceptions.WebSocketException as ws_error:
                logger.warning(f"WebSocket error: {ws_error}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
            
            if self._running:
                logger.info(f"Attempting reconnect in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_delay)
    
    async def _cleanup(self) -> None:
        """Cleanup async resources."""
        pass
    
    @staticmethod
    def _decode_payload(payload) -> Optional[np.ndarray]:
        """
        Decode incoming payload as numpy array.
        
        Args:
            payload: Raw payload from WebSocket (bytes or str).
        
        Returns:
            Decoded thermal frame or None if invalid.
        """
        try:
            if isinstance(payload, bytes):
                # Try NPY format first
                try:
                    with io.BytesIO(payload) as buffer:
                        arr = np.load(buffer, allow_pickle=False)
                    return np.asarray(arr)
                except Exception:
                    # Fall back to raw float32 binary
                    if len(payload) % 4 != 0:
                        logger.warning("Invalid binary payload size")
                        return None
                    
                    arr = np.frombuffer(payload, dtype="<f4")
                    if arr.size == THERMAL_WIDTH * THERMAL_HEIGHT:
                        return arr.reshape((THERMAL_HEIGHT, THERMAL_WIDTH))
                    return arr
            
            if isinstance(payload, str):
                try:
                    obj = json.loads(payload)
                    
                    # Handle base64 encoded NPY
                    if "npy_base64" in obj:
                        raw = base64.b64decode(obj["npy_base64"])
                        with io.BytesIO(raw) as buffer:
                            arr = np.load(buffer, allow_pickle=False)
                        return np.asarray(arr)
                    
                    # Handle base64 encoded float32
                    if "float32_base64" in obj:
                        raw = base64.b64decode(obj["float32_base64"])
                        if len(raw) % 4 != 0:
                            logger.warning("Invalid float32_base64 size")
                            return None
                        arr = np.frombuffer(raw, dtype="<f4")
                        if arr.size == THERMAL_WIDTH * THERMAL_HEIGHT:
                            return arr.reshape((THERMAL_HEIGHT, THERMAL_WIDTH))
                        return arr
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON payload")
                    return None
            
            return None
        except Exception as e:
            logger.error(f"Payload decode exception: {e}")
            return None
