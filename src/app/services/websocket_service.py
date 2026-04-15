"""WebSocket service for monitoring ESP32 thermal data streams."""

import asyncio
import base64
import io
import ipaddress
import json
import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, Callable, Iterable, Any

import numpy as np
import websockets

from ..models import db, ESPNode, Temperature


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


class ESPNetworkScanner:
    """Scan local network hosts that expose a WebSocket service on port 81."""

    def __init__(
        self,
        ws_port: int = 81,
        timeout_seconds: float = 0.35,
        max_workers: int = 64,
    ):
        self.ws_port = ws_port
        self.timeout_seconds = timeout_seconds
        self.max_workers = max_workers

    def discover(
        self,
        subnet_cidr: Optional[str] = None,
        candidates: Optional[Iterable[str]] = None,
    ) -> list[dict[str, str]]:
        """Return reachable hosts as websocket URIs."""
        hosts = set(candidates or [])
        hosts.update(self._hosts_from_subnet(subnet_cidr))
        hosts = {host for host in hosts if self._is_ipv4(host)}

        if not hosts:
            return []

        discovered: list[dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._probe_host, host): host for host in hosts}
            for future in as_completed(futures):
                host = futures[future]
                try:
                    if future.result():
                        logger.info("ESP32 discovered at ip=%s", host)
                        discovered.append({
                            "ip_address": host,
                            "ws_uri": f"ws://{host}:{self.ws_port}/",
                        })
                except Exception as error:
                    logger.debug("Probe error for %s: %s", host, error)

        discovered.sort(key=lambda item: item["ip_address"])
        return discovered

    def _probe_host(self, host: str) -> bool:
        """Check whether a host accepts TCP connection on ws port."""
        try:
            with socket.create_connection((host, self.ws_port), timeout=self.timeout_seconds):
                return True
        except OSError:
            return False

    def _hosts_from_subnet(self, subnet_cidr: Optional[str]) -> list[str]:
        """Generate host list from provided subnet or local /24 fallback."""
        subnet = subnet_cidr.strip() if subnet_cidr else ""

        if not subnet:
            subnets = self._resolve_local_subnets()
            hosts: list[str] = []
            for local_subnet in subnets:
                try:
                    network = ipaddress.ip_network(local_subnet, strict=False)
                    if network.version == 4:
                        hosts.extend(str(ip) for ip in network.hosts())
                except ValueError:
                    continue
            return hosts

        try:
            network = ipaddress.ip_network(subnet, strict=False)
        except ValueError:
            logger.warning("Invalid subnet for scan: %s", subnet)
            return []

        if network.version != 4:
            return []

        return [str(ip) for ip in network.hosts()]

    @staticmethod
    def _resolve_local_subnets() -> list[str]:
        """Best-effort local /24 subnet discovery from active IPv4 addresses."""
        subnets: set[str] = set()

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                current_ip = sock.getsockname()[0]
                if current_ip and not current_ip.startswith("127."):
                    subnets.add(f"{current_ip.rsplit('.', 1)[0]}.0/24")
        except OSError:
            pass

        try:
            host_name = socket.gethostname()
            host_ips = {
                item[4][0]
                for item in socket.getaddrinfo(host_name, None, socket.AF_INET)
                if item and item[4]
            }
            for ip_addr in host_ips:
                ip_addr = str(ip_addr)
                if not ip_addr.startswith("127."):
                    subnets.add(f"{ip_addr.rsplit('.', 1)[0]}.0/24")
        except OSError:
            pass

        return sorted(subnets)

    @staticmethod
    def _is_ipv4(host: str) -> bool:
        try:
            return isinstance(ipaddress.ip_address(host), ipaddress.IPv4Address)
        except ValueError:
            return False


class ESPFleetWebSocketService:
    """Track multiple ESP nodes, scan network, and persist temperatures periodically."""

    def __init__(
        self,
        app: Any,
        scanner: Optional[ESPNetworkScanner] = None,
        scan_subnet: Optional[str] = None,
        auto_scan_on_startup: bool = True,
        persist_interval_seconds: int = 900,
        flush_interval_seconds: int = 5,
    ):
        self.app = app
        self.scanner = scanner or ESPNetworkScanner()
        self.scan_subnet = scan_subnet
        self.auto_scan_on_startup = auto_scan_on_startup
        self.persist_interval_seconds = persist_interval_seconds
        self.flush_interval_seconds = flush_interval_seconds

        self._running = False
        self._lock = threading.Lock()
        self._flush_thread: Optional[threading.Thread] = None
        self._node_streams: dict[int, WebSocketService] = {}
        self._latest_temperatures: dict[int, float] = {}
        self._last_saved_at: dict[int, datetime] = {}
        self._last_discovery: list[dict[str, str]] = []

    def start(self) -> None:
        """Start monitoring existing nodes and periodic temperature flushing."""
        if self._running:
            raise RuntimeError("ESP fleet service already running")

        self._running = True

        if self.auto_scan_on_startup:
            try:
                self.scan_network(subnet_cidr=self.scan_subnet, register=False)
            except Exception as error:
                logger.warning("Startup network scan failed: %s", error)

        self._start_registered_nodes()

        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        logger.info("ESP fleet websocket service started")

    def stop(self) -> None:
        """Stop all node streams and background flush worker."""
        if not self._running:
            return

        self._running = False

        with self._lock:
            streams = list(self._node_streams.values())
            self._node_streams.clear()

        for stream in streams:
            try:
                stream.stop()
            except Exception as error:
                logger.debug("Error stopping node stream: %s", error)

        if self._flush_thread:
            self._flush_thread.join(timeout=5)

        logger.info("ESP fleet websocket service stopped")

    def scan_network(
        self,
        subnet_cidr: Optional[str] = None,
        candidates: Optional[Iterable[str]] = None,
        register: bool = False,
    ) -> list[dict[str, str]]:
        """Scan network and optionally register discovered hosts in DB."""
        merged_candidates = set(candidates or [])
        with self.app.app_context():
            for node in ESPNode.query.all():
                merged_candidates.add(node.ip_address)

        discovered = self.scanner.discover(
            subnet_cidr=subnet_cidr or self.scan_subnet,
            candidates=merged_candidates,
        )
        self._last_discovery = discovered

        if register and discovered:
            self.register_ips([item["ip_address"] for item in discovered])

        return discovered

    def get_last_discovery(self) -> list[dict[str, str]]:
        """Return cached discovery result from latest scan."""
        return list(self._last_discovery)

    def register_ips(self, ips: Iterable[str]) -> list[dict[str, Any]]:
        """Create missing ESP nodes and start stream tracking for each host."""
        cleaned_ips = sorted({ip.strip() for ip in ips if ip and ip.strip()})
        if not cleaned_ips:
            return []

        registered: list[dict[str, Any]] = []
        with self.app.app_context():
            for ip_address in cleaned_ips:
                node = ESPNode.query.filter_by(ip_address=ip_address).first()
                if node is None:
                    node = ESPNode(ip_address=ip_address)
                    db.session.add(node)
                    db.session.flush()
                    logger.info("Registered discovered ESP32 ip=%s node_uid=%s", node.ip_address, node.node_uid)

                registered.append(node.to_dict())
                self.track_node(node.id, node.ip_address)

            db.session.commit()

        return registered

    def track_node(self, node_id: int, ip_address: str) -> None:
        """Start websocket stream tracking for a single registered node."""
        with self._lock:
            if node_id in self._node_streams:
                return

            uri = f"ws://{ip_address}:81/"
            stream = WebSocketService(
                uri=uri,
                on_frame_callback=lambda frame, target_id=node_id: self._on_frame(target_id, frame),
            )
            self._node_streams[node_id] = stream

        if self._running:
            try:
                stream.start()
            except Exception as error:
                logger.warning("Unable to start stream for node %s: %s", ip_address, error)

    def untrack_node(self, node_id: int) -> None:
        """Stop tracking a node that has been removed from DB."""
        with self._lock:
            stream = self._node_streams.pop(node_id, None)
            self._latest_temperatures.pop(node_id, None)
            self._last_saved_at.pop(node_id, None)

        if stream:
            stream.stop()

    def _start_registered_nodes(self) -> None:
        """Attach stream monitoring to all nodes currently stored in DB."""
        with self.app.app_context():
            nodes = ESPNode.query.all()

        for node in nodes:
            self.track_node(node.id, node.ip_address)

    def _on_frame(self, node_id: int, frame: np.ndarray) -> None:
        """Store latest representative temperature from incoming frame."""
        try:
            arr = np.asarray(frame, dtype=np.float32)
            if arr.size == 0:
                return
            representative_temp = float(np.nanmean(arr))
        except Exception:
            return

        with self._lock:
            self._latest_temperatures[node_id] = representative_temp

    def _flush_loop(self) -> None:
        """Periodically persist latest temperatures for tracked nodes."""
        while self._running:
            try:
                self._flush_due_temperatures()
            except Exception as error:
                logger.warning("Temperature flush failed: %s", error)
            time.sleep(self.flush_interval_seconds)

    def _flush_due_temperatures(self, now: Optional[datetime] = None) -> int:
        """Persist temperatures for nodes that reached save interval."""
        now_dt = now or datetime.utcnow()

        with self._lock:
            snapshot = dict(self._latest_temperatures)
            last_saved = dict(self._last_saved_at)

        inserted = 0
        with self.app.app_context():
            for node_id, temperature in snapshot.items():
                previous_save = last_saved.get(node_id)
                if previous_save and (now_dt - previous_save).total_seconds() < self.persist_interval_seconds:
                    continue

                node = ESPNode.query.get(node_id)
                if node is None:
                    continue

                event_key = f"ws_{node_id}_{int(now_dt.timestamp())}"
                db.session.add(
                    Temperature(
                        esp_node_id=node_id,
                        event_key=event_key,
                        temperature=temperature,
                        measured_at=now_dt,
                    )
                )
                inserted += 1

            if inserted:
                db.session.commit()

        if inserted:
            with self._lock:
                for node_id in snapshot:
                    if node_id in self._latest_temperatures:
                        self._last_saved_at[node_id] = now_dt

        return inserted
