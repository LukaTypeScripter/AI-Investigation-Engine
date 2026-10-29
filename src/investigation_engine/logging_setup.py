import logging

PACKAGE_LOGGER = "investigation_engine"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
ELLIPSIS = "..."


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + ELLIPSIS
