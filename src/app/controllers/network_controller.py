"""Controllers for backend-side network discovery."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import HTTPConnection
import logging
from socket import AF_INET, SOCK_STREAM, create_connection, getaddrinfo, gethostname , socket , SOCK_DGRAM
from typing import Iterable
from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

network_bp = Blueprint("network", __name__, url_prefix="/api/network")

_ESP32_PAGE_SIGNATURES = (
    "Caméra Thermique 32x24",
    "Heatmap Thermique",
    "WebSocketsServer",
    "ws://",
)


def _normalize_hosts(values: Iterable[str]) -> list[str]:
    hosts: list[str] = []
    seen: set[str] = set()
    for value in values:
        host = str(value).strip()
        if not host or host in seen:
            continue
        seen.add(host)
        hosts.append(host)
    return hosts

def _local_listen_hosts() -> list[str]:
    """IPv4 locales sur lesquelles le serveur écoute, hors loopback (127.*)."""
    hosts: set[str] = set()

    # IP de sortie principale (route par défaut) — fiable même avec plusieurs interfaces
    try:
        s = socket(AF_INET, SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            hosts.add(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass

    # Toutes les IPv4 résolues par le hostname
    try:
        for info in getaddrinfo(gethostname(), None, AF_INET, SOCK_STREAM):
            hosts.add(info[4][0])
    except OSError:
        pass

    return _normalize_hosts(
        h for h in hosts if h and not h.startswith("127.")
    )


def _looks_like_esp32_page(body: str) -> bool:
    return any(signature in body for signature in _ESP32_PAGE_SIGNATURES)


def _probe_port_81(host: str, timeout_s: float) -> bool:
    try:
        sock = create_connection((host, 81), timeout=timeout_s)
        sock.close()
        return True
    except Exception:
        return False


def _probe_http_signature(host: str, timeout_s: float) -> bool:
    conn = None
    try:
        conn = HTTPConnection(host=host, port=80, timeout=timeout_s)
        conn.request("GET", "/")
        response = conn.getresponse()
        if response.status != 200:
            return False

        body = response.read(4096).decode("utf-8", errors="ignore")
        return _looks_like_esp32_page(body)
    except Exception:
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _is_esp32_host(host: str, timeout_s: float) -> bool:
    if _probe_port_81(host, timeout_s):
        return True
    return _probe_http_signature(host, timeout_s)


def _collect_ipv4_prefixes(seed_hosts: Iterable[str]) -> list[str]:
    prefixes: set[str] = set()

    for host in seed_hosts:
        parts = host.split(".")
        if len(parts) == 4 and all(part.isdigit() for part in parts):
            prefixes.add(".".join(parts[:3]))

    try:
        for family, socktype, _, _, sockaddr in getaddrinfo(gethostname(), None):
            if family != AF_INET or socktype != SOCK_STREAM:
                continue
            ip = sockaddr[0]
            parts = ip.split(".")
            if len(parts) == 4 and not ip.startswith("127."):
                prefixes.add(".".join(parts[:3]))
    except Exception:
        pass

    return sorted(prefixes)


def _scan_hosts(hosts: list[str], timeout_s: float, workers: int, max_results: int) -> list[str]:
    if not hosts:
        return []

    discovered: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_is_esp32_host, host, timeout_s): host for host in hosts}
        for future in as_completed(futures):
            host = futures[future]
            try:
                if future.result():
                    discovered.append(host)
                    if len(discovered) >= max_results:
                        break
            except Exception:
                continue

    return discovered


@network_bp.route("/scan-esp32", methods=["POST"])
def scan_esp32_on_backend():
    """Scan local network from backend host and return detected ESP32 hosts."""
    data = request.get_json(silent=True) or {}

    preferred_hosts = _normalize_hosts(data.get("preferred_hosts", []))
    extra_candidates = _normalize_hosts(data.get("extra_candidates", []))
    server_hosts = _local_listen_hosts()
    logger.info("Hôtes d'écoute du backend : %s", server_hosts)
    extra_candidates = _normalize_hosts([*extra_candidates, *server_hosts])


    timeout_ms = int(data.get("timeout_ms", 700))
    timeout_ms = max(150, min(timeout_ms, 5000))
    timeout_s = timeout_ms / 1000.0

    max_results = int(data.get("max_results", 5))
    max_results = max(1, min(max_results, 20))

    workers = int(data.get("workers", 48))
    workers = max(4, min(workers, 128))

    scan_full_subnet = bool(data.get("scan_full_subnet", True))

    candidate_hosts = _normalize_hosts([*preferred_hosts, *extra_candidates])
    discovered_hosts = _scan_hosts(candidate_hosts, timeout_s, workers, max_results)

    if scan_full_subnet and len(discovered_hosts) < max_results:
        excluded = set(candidate_hosts)
        prefixes = _collect_ipv4_prefixes(candidate_hosts)

        for prefix in prefixes:
            subnet_hosts = [
                f"{prefix}.{host_index}"
                for host_index in range(1, 255)
                if f"{prefix}.{host_index}" not in excluded
            ]

            remaining = max_results - len(discovered_hosts)
            found = _scan_hosts(subnet_hosts, timeout_s, workers, remaining)
            for host in found:
                if host not in discovered_hosts:
                    discovered_hosts.append(host)

            if len(discovered_hosts) >= max_results:
                break

    return jsonify(
        {
            "discovered_hosts": discovered_hosts,
            "first_host": discovered_hosts[0] if discovered_hosts else None,
            "scanned_candidate_count": len(candidate_hosts),
            "timeout_ms": timeout_ms,
        }
    ), 200
