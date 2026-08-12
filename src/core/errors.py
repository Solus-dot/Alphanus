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


class ConfigurationError(AlphanusError, ValueError):
    code = "E_CONFIG"


class ProviderError(AlphanusError):
    code = "E_PROVIDER"


class ToolRuntimeError(AlphanusError, RuntimeError):
    code = "E_TOOL"


class ProtocolError(AlphanusError, ValueError):
    code = "E_PROTOCOL"


class OperationCancelled(AlphanusError):
    code = "E_CANCELLED"
