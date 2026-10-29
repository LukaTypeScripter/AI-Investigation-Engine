class LlmError(Exception):
    pass


class TransientLlmError(LlmError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LlmRateLimited(TransientLlmError):
    pass


class LlmUnavailable(TransientLlmError):
    pass


class LlmBadGeneration(TransientLlmError):
    pass


class LlmRejected(LlmError):
    pass


class LlmMalformedResponse(LlmError):
    pass
