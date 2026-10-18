import logging
import time

from kb_builder.context import SiteContext
from kb_builder.models import CrawlConfig, ExtractConfig, SegmentConfig, StageResult
from kb_builder.services import BaseService, CrawlService, ExtractService, SegmentService

logger = logging.getLogger("kb_builder.pipeline")


class PipelineError(RuntimeError):
    pass


class Pipeline:
    """Run stages in order; each reads what the previous one wrote to disk."""

    def __init__(self, stages: list[BaseService]) -> None:
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate stage names: {names}")
        self.stages = stages

    @property
    def names(self) -> list[str]:
        return [stage.name for stage in self.stages]

    def run(
        self,
        context: SiteContext,
        start: str | None = None,
        stop: str | None = None,
    ) -> dict[str, StageResult]:
        """Run stages from `start` to `stop` (both inclusive, default: all)."""
        results: dict[str, StageResult] = {}
        for stage in self._select(start, stop):
            missing = [path for path in stage.inputs(context) if not path.exists()]
            if missing:
                raise PipelineError(
                    f"Stage '{stage.name}' needs {', '.join(map(str, missing))}; run the earlier stages first."
                )

            logger.info("\n=== %s: %s", stage.name, context.site)
            started = time.perf_counter()
            result = stage.run(context)
            elapsed = time.perf_counter() - started

            counts = ", ".join(f"{key}: {n}" for key, n in result.counts().items())
            logger.info("=== %s done in %.1fs -> %s  (%s)", stage.name, elapsed, result.out_dir, counts)
            results[stage.name] = result
        return results

    def _select(self, start: str | None, stop: str | None) -> list[BaseService]:
        for name in (start, stop):
            if name is not None and name not in self.names:
                raise PipelineError(f"Unknown stage '{name}'. Stages: {', '.join(self.names)}")
        first = self.names.index(start) if start else 0
        last = self.names.index(stop) if stop else len(self.stages) - 1
        if first > last:
            raise PipelineError(f"Stage '{start}' comes after '{stop}'.")
        return self.stages[first : last + 1]


def build_pipeline(
    crawl: CrawlConfig | None = None,
    extract: ExtractConfig | None = None,
    segment: SegmentConfig | None = None,
) -> Pipeline:
    return Pipeline([CrawlService(crawl), ExtractService(extract), SegmentService(segment)])
