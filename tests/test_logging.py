"""Tests for utils/logging.py — setup_logger behaviour."""
import logging


from kneevision.utils.logging import setup_logger


# ── setup_logger ───────────────────────────────────────────────────────────────

def test_setup_logger_returns_logger():
    logger = setup_logger("test_kv_logger_basic")
    assert isinstance(logger, logging.Logger)


def test_setup_logger_correct_name():
    logger = setup_logger("test_kv_logger_name")
    assert logger.name == "test_kv_logger_name"


def test_setup_logger_default_level_is_info():
    logger = setup_logger("test_kv_logger_level_info")
    assert logger.level == logging.INFO


def test_setup_logger_custom_level():
    logger = setup_logger("test_kv_logger_debug", level=logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_setup_logger_has_stream_handler():
    logger = setup_logger("test_kv_logger_stream")
    handler_types = [type(h) for h in logger.handlers]
    assert logging.StreamHandler in handler_types


def test_setup_logger_has_file_handler():
    logger = setup_logger("test_kv_logger_file")
    handler_types = [type(h) for h in logger.handlers]
    assert logging.FileHandler in handler_types


def test_setup_logger_idempotent():
    """Calling setup_logger twice with the same name should return same handlers (no duplication)."""
    name = "test_kv_logger_idempotent"
    logger1 = setup_logger(name)
    n_handlers = len(logger1.handlers)
    logger2 = setup_logger(name)
    assert len(logger2.handlers) == n_handlers, "handlers should not be added twice"
    assert logger1 is logger2


def test_setup_logger_creates_log_dir():
    """setup_logger should create the logs/ directory automatically."""
    from kneevision.config.settings import PROJECT_ROOT
    setup_logger("test_kv_logger_dir")
    assert (PROJECT_ROOT / "logs").is_dir()


def test_setup_logger_log_file_created():
    """A log file for the given name should exist after setup."""
    from kneevision.config.settings import PROJECT_ROOT
    name = "test_kv_logger_file_exists"
    setup_logger(name)
    log_file = PROJECT_ROOT / "logs" / f"{name}.log"
    assert log_file.exists()


def test_setup_logger_can_log_messages(capfd):
    """Logger should emit INFO messages to stdout."""
    logger = setup_logger("test_kv_logger_emit")
    logger.info("unit test message")
    captured = capfd.readouterr()
    assert "unit test message" in captured.out


def test_setup_logger_format_contains_level(capfd):
    """Log format should include the level name."""
    logger = setup_logger("test_kv_logger_format")
    logger.warning("format check")
    captured = capfd.readouterr()
    assert "WARNING" in captured.out
