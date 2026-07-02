from typing import Optional

from loguru import logger as eval_logger

from ..base import ServerInterface
from ..protocol import Request, Response, ServerConfig


class DummyProvider(ServerInterface):
    """No-op judge: every evaluation returns the literal string "dummy".

    Binary judge metrics parse this as 0, so any llm_as_judge score produced
    under this provider is a placeholder, not a model score.
    """

    _warned = False

    def __init__(self, config: Optional[ServerConfig] = None):
        super().__init__(config)

    def is_available(self) -> bool:
        return True

    def evaluate(self, request: Request) -> Response:
        # Warn on first actual use, not on construction: task modules build
        # judge servers at import time even when scoring never calls them.
        if not DummyProvider._warned:
            DummyProvider._warned = True
            eval_logger.warning("DummyProvider is SCORING responses (API_TYPE=dummy): these llm_as_judge metrics are placeholder values, NOT real scores. Wire a judge endpoint or exclude judge-scored tasks.")
        dummy_response = Response(content="dummy", model_used="dummy", usage="dummy", raw_response="dummy")
        return dummy_response
