import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from kb_builder.context import SiteContext
from kb_builder.models import StageResult


class BaseService[ResultT: StageResult](ABC):
    """One pipeline stage: configured in `__init__`, executed by `run(context)`."""

    name: ClassVar[str]

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"kb_builder.{type(self).__name__}")

    def inputs(self, context: SiteContext) -> list[Path]:
        """Files written by earlier stages that this one reads."""
        return []

    @abstractmethod
    def run(self, context: SiteContext) -> ResultT: ...
