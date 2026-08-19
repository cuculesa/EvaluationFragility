from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_id(prefix: str, text: str, length: int = 16) -> str:
    return f"{prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:length]}"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json_object(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return cast(dict[str, Any], value)


def atomic_write_bytes(path: Path, data: bytes, mode: int = 0o640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path, text: str, mode: int = 0o640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def git_revision(cwd: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL, timeout=3
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def runtime_fingerprint() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "token",
    "password",
    "secret",
    "credential",
    "authorization",
    "cookie",
)


def _redact_url_credentials(value: str) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if not parts.scheme or not parts.netloc:
        return value
    netloc = parts.netloc
    if parts.username is not None:
        host = parts.hostname or ""
        if parts.port is not None:
            host = f"{host}:{parts.port}"
        netloc = f"<redacted>@{host}"
    query = "<redacted>" if parts.query else ""
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def redact_secrets(value: Any) -> Any:
    """Redact likely credentials before writing configuration/provenance artifacts."""
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            secret_key = any(marker in normalized for marker in _SECRET_MARKERS)
            secret_key = secret_key or normalized.endswith("_key")
            if secret_key:
                redacted[key] = "<redacted>"
            elif isinstance(item, str) and "url" in normalized:
                redacted[key] = _redact_url_credentials(item)
            else:
                redacted[key] = redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def build_file_manifest(root: Path, *, exclude: set[str] | None = None) -> dict[str, Any]:
    """Hash regular files below ``root`` using stable relative paths."""
    excluded = exclude or set()
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        files[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {"schema_version": 1, "files": files}
