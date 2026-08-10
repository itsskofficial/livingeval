"""Where records, figures and datasets go when the process is not your laptop.

A container's filesystem disappears when the task is replaced. Everything this library
treats as an artifact — result records, figures, exported fine-tuning data — has to land
somewhere that outlives it, or the deployed service quietly stops producing anything you
can look at afterwards.

Three backends behind one interface, chosen by URL:

| URL | backend | for |
|---|---|---|
| `file://./results` | local filesystem | your laptop, and the default |
| `s3://bucket/prefix` | AWS S3 | an ECS/Fargate deployment |
| `supabase://bucket/prefix` | Supabase Storage | when Supabase is already your database |

`boto3` and the Supabase REST call are both lazy, so neither is a dependency of the
default install.

**Artifacts are immutable and content-addressed by name.** `put("audit.json", …)`
overwrites; `put_versioned` writes `audit/2026-08-09T12-00-00Z.json` and updates
`audit/latest.json`. The deployed worker uses the versioned form, so a history
accumulates and the dashboard always has a `latest` to read — which is what you want
when the interesting question is usually "when did this number change".
"""

from __future__ import annotations

import io
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "LocalArtifacts",
    "S3Artifacts",
    "SupabaseArtifacts",
    "open_artifacts",
]


@dataclass(frozen=True)
class ArtifactRef:
    """Where something was written and how to find it again."""

    uri: str
    key: str
    size: int
    backend: str

    def as_dict(self) -> dict:
        return {"uri": self.uri, "key": self.key, "size": self.size, "backend": self.backend}


@runtime_checkable
class ArtifactStore(Protocol):
    backend: str
    base: str

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> ArtifactRef: ...
    def get_bytes(self, key: str) -> bytes: ...
    def list(self, prefix: str = "") -> list[str]: ...
    def exists(self, key: str) -> bool: ...
    def uri(self, key: str) -> str: ...


def _version_key() -> str:
    """A sortable UTC key that is unique even for writes in the same instant.

    Two problems, both of which lose history silently because an object store
    overwrites rather than errors:

    - Seconds are too coarse. Two ticks, or a manual re-run, land on the same key.
    - Milliseconds are *also* not enough on Windows, where the system clock ticks
      about every 16 ms, so several consecutive calls genuinely return the same value.

    So the key is a millisecond timestamp plus six random hex characters. The timestamp
    prefix keeps the keys sorting chronologically; the suffix makes a collision a
    non-event rather than a lost version.
    """
    now = time.time()
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime(now))
    return f"{stamp}.{int(now % 1 * 1000):03d}Z-{secrets.token_hex(3)}"


class _Common:
    """Shared conveniences layered over the byte-level interface."""

    def put(self, key: str, payload: Any, content_type: str | None = None) -> ArtifactRef:
        """Write JSON, text or bytes, inferred from the payload."""
        if isinstance(payload, bytes):
            return self.put_bytes(key, payload, content_type or "application/octet-stream")  # type: ignore[attr-defined]
        if isinstance(payload, str):
            return self.put_bytes(key, payload.encode("utf-8"), content_type or "text/plain; charset=utf-8")  # type: ignore[attr-defined]
        data = json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        return self.put_bytes(key, data, content_type or "application/json")  # type: ignore[attr-defined]

    def put_versioned(self, name: str, payload: Any, content_type: str | None = None) -> dict:
        """Write `<name>/<timestamp><ext>` and refresh `<name>/latest<ext>`.

        Two writes rather than one, because a deployed service wants both a history and
        a stable URL, and object stores have no rename.
        """
        stem, _, ext = name.rpartition(".")
        stem, ext = (stem or name), (f".{ext}" if stem else "")
        versioned = self.put(f"{stem}/{_version_key()}{ext}", payload, content_type)
        latest = self.put(f"{stem}/latest{ext}", payload, content_type)
        return {"versioned": versioned.as_dict(), "latest": latest.as_dict()}

    def get_json(self, key: str) -> Any:
        return json.loads(self.get_bytes(key).decode("utf-8"))  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# local
# ---------------------------------------------------------------------------


class LocalArtifacts(_Common):
    """The filesystem. The default, and the right answer on a laptop."""

    backend = "file"

    def __init__(self, root: str | Path = "./results"):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.base = f"file://{self.root}"

    def _path(self, key: str) -> Path:
        # Refuse to escape the root. `key` can come from an API request.
        target = (self.root / key.lstrip("/")).resolve()
        if not str(target).startswith(str(self.root)):
            raise ValueError(f"artifact key {key!r} escapes the artifact root")
        return target

    def put_bytes(self, key: str, data: bytes, content_type: str = "") -> ArtifactRef:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return ArtifactRef(self.uri(key), key, len(data), self.backend)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list(self, prefix: str = "") -> list[str]:
        base = self._path(prefix) if prefix else self.root
        if not base.exists():
            return []
        if base.is_file():
            return [str(base.relative_to(self.root)).replace(os.sep, "/")]
        return sorted(
            str(p.relative_to(self.root)).replace(os.sep, "/")
            for p in base.rglob("*") if p.is_file()
        )

    def uri(self, key: str) -> str:
        return f"file://{self._path(key)}"


# ---------------------------------------------------------------------------
# s3
# ---------------------------------------------------------------------------


class S3Artifacts(_Common):
    """AWS S3. Credentials come from the standard chain — on ECS that is the task role,
    which is why nothing here takes an access key."""

    backend = "s3"

    def __init__(self, bucket: str, prefix: str = "", region: str | None = None):
        try:
            import boto3
        except ImportError as e:  # pragma: no cover - optional extra
            raise ImportError("pip install 'livingeval[aws]'") from e
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        self._s3 = boto3.client("s3", region_name=self.region)
        self.base = f"s3://{bucket}/{self.prefix}".rstrip("/")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key.lstrip('/')}" if self.prefix else key.lstrip("/")

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> ArtifactRef:  # pragma: no cover - network
        self._s3.put_object(Bucket=self.bucket, Key=self._key(key), Body=data,
                            ContentType=content_type)
        return ArtifactRef(self.uri(key), key, len(data), self.backend)

    def get_bytes(self, key: str) -> bytes:  # pragma: no cover - network
        buf = io.BytesIO()
        self._s3.download_fileobj(self.bucket, self._key(key), buf)
        return buf.getvalue()

    def exists(self, key: str) -> bool:  # pragma: no cover - network
        from botocore.exceptions import ClientError

        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except ClientError:
            return False

    def list(self, prefix: str = "") -> list[str]:  # pragma: no cover - network
        out: list[str] = []
        token = None
        full = self._key(prefix)
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": full}
            if token:
                kwargs["ContinuationToken"] = token
            page = self._s3.list_objects_v2(**kwargs)
            for obj in page.get("Contents", []):
                key = obj["Key"]
                out.append(key[len(self.prefix) + 1:] if self.prefix else key)
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")
        return sorted(out)

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._key(key)}"

    def presign(self, key: str, expires_s: int = 3600) -> str:  # pragma: no cover - network
        """A time-limited URL, so a figure can be shown without making the bucket public."""
        return self._s3.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": self._key(key)},
            ExpiresIn=expires_s,
        )


# ---------------------------------------------------------------------------
# supabase storage
# ---------------------------------------------------------------------------


class SupabaseArtifacts(_Common):
    """Supabase Storage over its REST API.

    Uses `urllib` rather than the `supabase` SDK: the storage API is three HTTP calls,
    and the SDK would be a dependency carrying an httpx pin for no benefit.

    Needs `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`. The service-role key bypasses
    row-level security, which is correct for a backend writing its own artifacts and
    wrong for anything that reaches a browser — so it is read from the environment only,
    and never returned by `/v1/status`.
    """

    backend = "supabase"

    def __init__(self, bucket: str, prefix: str = "", url: str | None = None,
                 service_key: str | None = None):
        self.url = (url or os.environ.get("SUPABASE_URL", "")).rstrip("/")
        self.key = service_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not self.url or not self.key:
            raise RuntimeError(
                "Supabase Storage needs SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the "
                "environment. Both are in your project's API settings; the service-role "
                "key is the secret one and must never reach a browser."
            )
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.base = f"supabase://{bucket}/{self.prefix}".rstrip("/")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key.lstrip('/')}" if self.prefix else key.lstrip("/")

    def _request(self, method: str, path: str, data: bytes | None = None,
                 content_type: str | None = None,
                 retries: int = 4) -> bytes:  # pragma: no cover - network
        """One Storage call, retrying only the gateway errors that are transient.

        502/503/504 from Supabase's gateway means the storage service is still coming
        up — `supabase start` returns once the containers exist, not once they answer.
        Retrying those is correct; retrying a 4xx would just repeat a mistake, so it
        does not.
        """
        import time as _time
        import urllib.error
        import urllib.request

        last: Exception | None = None
        for attempt in range(1, retries + 1):
            req = urllib.request.Request(f"{self.url}/storage/v1/{path}", data=data,
                                         method=method)
            req.add_header("Authorization", f"Bearer {self.key}")
            req.add_header("apikey", self.key)
            if content_type:
                req.add_header("Content-Type", content_type)
                # Supabase rejects a repeat upload without this, which reads as a
                # permissions error and is not one.
                req.add_header("x-upsert", "true")
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return resp.read()
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:300]
                if e.code in (502, 503, 504) and attempt < retries:
                    last = e
                    _time.sleep(1.5 * attempt)
                    continue
                raise RuntimeError(
                    f"supabase storage {method} {path} -> {e.code}: {body}"
                ) from e
            except urllib.error.URLError as e:
                if attempt < retries:
                    last = e
                    _time.sleep(1.5 * attempt)
                    continue
                raise RuntimeError(
                    f"cannot reach supabase storage at {self.url}: {e.reason}"
                ) from e
        raise RuntimeError(f"supabase storage {method} {path} failed: {last}")

    def wait_ready(self, timeout_s: float = 60.0) -> bool:  # pragma: no cover - network
        """Block until the Storage API answers, or give up.

        Called before `supabase init` touches Storage. Without it the very first run
        after `supabase start` fails on a 502 that looks like a configuration error and
        is really just a service that has not finished booting.
        """
        import time as _time
        import urllib.error
        import urllib.request

        deadline = _time.time() + timeout_s
        while _time.time() < deadline:
            try:
                req = urllib.request.Request(f"{self.url}/storage/v1/bucket")
                req.add_header("Authorization", f"Bearer {self.key}")
                req.add_header("apikey", self.key)
                with urllib.request.urlopen(req, timeout=5):
                    return True
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    return True  # answering, even if it is refusing us
            except OSError:
                pass
            _time.sleep(2.0)
        return False

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> ArtifactRef:  # pragma: no cover - network
        self._request("POST", f"object/{self.bucket}/{self._key(key)}", data, content_type)
        return ArtifactRef(self.uri(key), key, len(data), self.backend)

    def get_bytes(self, key: str) -> bytes:  # pragma: no cover - network
        return self._request("GET", f"object/{self.bucket}/{self._key(key)}")

    def exists(self, key: str) -> bool:  # pragma: no cover - network
        try:
            self._request("GET", f"object/info/{self.bucket}/{self._key(key)}")
            return True
        except RuntimeError:
            return False

    def list(self, prefix: str = "") -> list[str]:  # pragma: no cover - network
        body = json.dumps({"prefix": self._key(prefix), "limit": 1000}).encode()
        raw = self._request("POST", f"object/list/{self.bucket}", body, "application/json")
        return sorted(item["name"] for item in json.loads(raw))

    def uri(self, key: str) -> str:
        return f"supabase://{self.bucket}/{self._key(key)}"

    # -- bucket management -----------------------------------------------------

    def bucket_exists(self) -> bool:  # pragma: no cover - network
        try:
            self._request("GET", f"bucket/{self.bucket}")
            return True
        except RuntimeError:
            return False

    def create_bucket(self, public: bool = False) -> dict:  # pragma: no cover - network
        """Create the bucket if it is missing. Private by default.

        Artifacts are result records and figures derived from production traces. A
        public bucket makes every one of them world-readable by URL, which is a data
        leak wearing a convenience label - so `public=False` and reads go through the
        service role or a signed URL.
        """
        if self.bucket_exists():
            return {"bucket": self.bucket, "created": False, "public": None}
        body = json.dumps(
            {"name": self.bucket, "id": self.bucket, "public": public}
        ).encode()
        self._request("POST", "bucket", body, "application/json")
        return {"bucket": self.bucket, "created": True, "public": public}

    def signed_url(self, key: str, expires_s: int = 3600) -> str:  # pragma: no cover - network
        """A time-limited URL, so a figure can be shown without the bucket being public."""
        body = json.dumps({"expiresIn": expires_s}).encode()
        raw = self._request("POST", f"object/sign/{self.bucket}/{self._key(key)}",
                            body, "application/json")
        return f"{self.url}/storage/v1{json.loads(raw)['signedURL']}"


# ---------------------------------------------------------------------------


def open_artifacts(url: str = "file://./results") -> ArtifactStore:
    """Open an artifact store from a URL.

    `file://./results`, `s3://bucket/prefix`, `supabase://bucket/prefix`.
    """
    parsed = urlparse(url)
    if parsed.scheme in ("", "file"):
        root = (parsed.netloc or "") + (parsed.path or "")
        return LocalArtifacts(root or "./results")  # type: ignore[return-value]
    if parsed.scheme == "s3":
        return S3Artifacts(parsed.netloc, parsed.path.lstrip("/"))  # type: ignore[return-value]
    if parsed.scheme == "supabase":
        return SupabaseArtifacts(parsed.netloc, parsed.path.lstrip("/"))  # type: ignore[return-value]
    raise ValueError(
        f"unrecognised artifact URL {url!r}; use file://, s3:// or supabase://"
    )
