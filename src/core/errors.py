from __future__ import annotations


class AlphanusError(Exception):
    code = "E_INTERNAL"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(str(message))
        self.__cause__ = cause


class OperationCancelled(AlphanusError):
    code = "E_CANCELLED"
