import logging
from abc import ABC, abstractmethod


class BaseService[ResultT](ABC):
    """One pipeline stage: configured in `__init__`, executed by `run()`."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"kb_builder.{type(self).__name__}")

    @abstractmethod
    def run(self) -> ResultT: ...
