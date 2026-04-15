"""Unit tests for ESP fleet websocket service and scanner."""

from datetime import datetime, timedelta

import numpy as np

from src.app.models import db, ESPNode, Temperature
from src.app.services.websocket_service import ESPNetworkScanner, ESPFleetWebSocketService


class StubScanner:
    """Deterministic scanner for unit tests."""

    def __init__(self, nodes):
        self.nodes = nodes

    def discover(self, subnet_cidr=None, candidates=None):
        return list(self.nodes)


class TestESPNetworkScanner:
    """Test subnet generation and host probing behavior."""

    def test_hosts_from_subnet_invalid(self):
        scanner = ESPNetworkScanner()
        hosts = scanner._hosts_from_subnet("not-a-subnet")
        assert hosts == []

    def test_discovery_logs_each_found_ip(self, monkeypatch, caplog):
        scanner = ESPNetworkScanner()

        monkeypatch.setattr(
            scanner,
            "_hosts_from_subnet",
            lambda _subnet: ["10.0.0.40", "10.0.0.41"],
        )
        monkeypatch.setattr(
            scanner,
            "_probe_host",
            lambda host: host == "10.0.0.41",
        )

        caplog.set_level("INFO")
        nodes = scanner.discover(subnet_cidr="10.0.0.0/24")
        assert [n["ip_address"] for n in nodes] == ["10.0.0.41"]
        assert "ESP32 discovered at ip=10.0.0.41" in caplog.text


class TestESPFleetWebSocketService:
    """Test multi-node registration and periodic persistence."""

    def test_register_ips_creates_nodes(self, app):
        service = ESPFleetWebSocketService(
            app=app,
            scanner=StubScanner([]),
            auto_scan_on_startup=False,
        )

        registered = service.register_ips(["10.0.0.10", "10.0.0.11"])
        assert len(registered) == 2

        with app.app_context():
            assert ESPNode.query.count() == 2

    def test_scan_network_registers_when_requested(self, app):
        service = ESPFleetWebSocketService(
            app=app,
            scanner=StubScanner([
                {"ip_address": "10.0.0.20", "ws_uri": "ws://10.0.0.20:81/"},
                {"ip_address": "10.0.0.21", "ws_uri": "ws://10.0.0.21:81/"},
            ]),
            auto_scan_on_startup=False,
        )

        nodes = service.scan_network(register=True)
        assert len(nodes) == 2

        with app.app_context():
            ips = {node.ip_address for node in ESPNode.query.all()}
            assert ips == {"10.0.0.20", "10.0.0.21"}

    def test_scan_network_merges_registered_ips(self, app):
        with app.app_context():
            db.session.add(ESPNode(ip_address="10.0.0.99", room_name="Saved"))
            db.session.commit()

        scanner_calls = []

        class CaptureScanner:
            def discover(self, subnet_cidr=None, candidates=None):
                scanner_calls.append(set(candidates or []))
                return []

        service = ESPFleetWebSocketService(
            app=app,
            scanner=CaptureScanner(),
            auto_scan_on_startup=False,
        )

        service.scan_network(candidates=["10.0.0.10"], register=False)
        assert scanner_calls
        assert "10.0.0.10" in scanner_calls[0]
        assert "10.0.0.99" in scanner_calls[0]

    def test_flush_due_temperatures_every_interval(self, app):
        service = ESPFleetWebSocketService(
            app=app,
            scanner=StubScanner([]),
            auto_scan_on_startup=False,
            persist_interval_seconds=900,
        )

        with app.app_context():
            node = ESPNode(ip_address="10.0.0.30", room_name="Lab")
            db.session.add(node)
            db.session.commit()
            node_id = node.id

        frame = np.ones((24, 32), dtype=np.float32) * 22.5
        service._on_frame(node_id, frame)

        t0 = datetime.utcnow()
        inserted = service._flush_due_temperatures(now=t0)
        assert inserted == 1

        inserted = service._flush_due_temperatures(now=t0 + timedelta(minutes=5))
        assert inserted == 0

        inserted = service._flush_due_temperatures(now=t0 + timedelta(minutes=16))
        assert inserted == 1

        with app.app_context():
            rows = Temperature.query.filter_by(esp_node_id=node_id).all()
            assert len(rows) == 2
