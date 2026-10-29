import logging

import pytest

from investigation_engine.app.main import create_default_app
from investigation_engine.logging_setup import PACKAGE_LOGGER, configure_logging, shorten

APP_LOGGER = f"{PACKAGE_LOGGER}.app.api"


def test_messages_at_or_above_the_level_are_written_to_standard_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    logger = logging.getLogger(APP_LOGGER)

    logger.info("engine ready")
    logger.debug("too detailed")

    written = capsys.readouterr().err
    assert "INFO investigation_engine.app.api: engine ready" in written
    assert "too detailed" not in written


def test_a_lower_level_shows_debug_messages(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("DEBUG")

    logging.getLogger(APP_LOGGER).debug("too detailed")

    assert "DEBUG investigation_engine.app.api: too detailed" in capsys.readouterr().err


def test_configuring_twice_does_not_repeat_messages(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    configure_logging("INFO")

    logging.getLogger(APP_LOGGER).info("only once")

    assert capsys.readouterr().err.count("only once") == 1


def test_shorten_keeps_short_text_and_cuts_long_text_with_an_ellipsis() -> None:
    assert shorten("short", 10) == "short"
    assert shorten("a" * 11, 10) == "a" * 10 + "..."


def test_the_default_app_configures_logging_from_the_log_level_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    create_default_app()

    assert logging.getLogger(PACKAGE_LOGGER).level == logging.DEBUG
