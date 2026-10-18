from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel

from kb_builder.urls import normalize, site_key


class SiteContext(BaseModel):
    """Which site the pipeline works on and where each stage keeps its data."""

    site: str
    start_url: str | None = None
    data_root: Path = Path("data")
    # The finished knowledge base lives outside the intermediate data folder.
    kb_root: Path = Path("knowledge_base")

    @classmethod
    def from_url(cls, url: str, data_root: Path = Path("data")) -> "SiteContext":
        start = normalize(url)
        if not start:
            raise ValueError(f"Not an http(s) URL: {url}")
        return cls(site=site_key(urlsplit(start).netloc), start_url=start, data_root=data_root)

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw" / self.site

    @property
    def extracted_dir(self) -> Path:
        return self.data_root / "extracted" / self.site

    @property
    def segmented_dir(self) -> Path:
        return self.data_root / "segmented" / self.site

    @property
    def profile_dir(self) -> Path:
        return self.data_root / "profile" / self.site

    @property
    def classified_dir(self) -> Path:
        return self.data_root / "classified" / self.site

    @property
    def merged_dir(self) -> Path:
        return self.data_root / "merged" / self.site

    @property
    def normalized_dir(self) -> Path:
        return self.data_root / "normalized" / self.site

    @property
    def kb_dir(self) -> Path:
        return self.kb_root / self.site
