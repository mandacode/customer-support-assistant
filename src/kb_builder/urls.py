import hashlib
import re
from urllib.parse import urljoin, urlsplit, urlunsplit

# Links to these are never pages worth fetching.
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".avif",
    ".css", ".js", ".json", ".xml", ".txt",
    ".mp4", ".webm", ".mp3", ".wav",
    ".zip", ".rar", ".7z",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
}

DOCUMENT_EXTENSIONS = {".pdf"}


def site_key(host: str) -> str:
    """Host without the `www.` prefix, so `www.x.com` and `x.com` are one site."""
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def normalize(url: str, base: str | None = None) -> str | None:
    """Absolute http(s) URL without fragment and query, or None if unusable.

    Query strings are dropped: on small company sites they are almost always
    tracking or filters, not separate content.
    """
    if base:
        url = urljoin(base, url)
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    path = re.sub(r"/{2,}", "/", parts.path) or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc.lower(), path, "", ""))


def extension(url: str) -> str:
    path = urlsplit(url).path.lower()
    match = re.search(r"\.[a-z0-9]{1,5}$", path)
    return match.group(0) if match else ""


def is_same_site(url: str, site: str) -> bool:
    return site_key(urlsplit(url).netloc) == site


def cache_name(url: str) -> str:
    """Readable, collision-safe file name for a URL, without extension."""
    path = urlsplit(url).path.strip("/")
    path = re.sub(r"\.[a-z0-9]{1,5}$", "", path, flags=re.I)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_").lower() or "index"
    digest = hashlib.sha1(url.encode()).hexdigest()[:8]
    return f"{slug[:80]}-{digest}"
