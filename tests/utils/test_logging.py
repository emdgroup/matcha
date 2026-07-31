"""Tests for matcha.utils.logging – get_default_logger and MatchaLogger.

Validates the matcha-specific logger factory (handler setup, file logging,
idempotent handler attachment) and the MatchaLogger property / method
overrides.  Does *not* re-test the standard-library logging module or
MLflow internals.
"""

import logging
import os
from unittest.mock import patch

import pytest

from matcha.utils.logging import MatchaLogger, get_default_logger


# =========================================================================
# get_default_logger
# =========================================================================


class TestGetDefaultLogger:
    """Tests for the get_default_logger factory."""

    def test_returns_logger_instance(self):
        logger = get_default_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert "MATCHA test_module" in logger.name

    def test_has_console_handler(self):
        logger = get_default_logger("console_test")
        handler_types = [type(h) for h in logger.handlers]
        assert logging.StreamHandler in handler_types

    def test_file_handler_when_path_given(self, tmp_path):
        log_file = str(tmp_path / "logs" / "test.log")
        logger = get_default_logger("file_test", logging_path=log_file)

        handler_types = [type(h) for h in logger.handlers]
        assert logging.FileHandler in handler_types
        assert os.path.isfile(log_file) or os.path.exists(os.path.dirname(log_file))

    def test_no_duplicate_handlers_on_repeat_call(self):
        """Calling get_default_logger twice for the same name should not
        add duplicate handlers."""
        name = "dedup_test"
        logger1 = get_default_logger(name)
        n_handlers_first = len(logger1.handlers)
        logger2 = get_default_logger(name)
        assert len(logger2.handlers) == n_handlers_first
        assert logger1 is logger2

    def test_log_level_is_info(self):
        logger = get_default_logger("level_test")
        assert logger.level == logging.INFO


# =========================================================================
# MatchaLogger
# =========================================================================


class TestMatchaLogger:
    """Tests for MatchaLogger property / method overrides."""

    @patch("matcha.utils.logging.MLFlowLogger.__init__", return_value=None)
    def test_store_property_default_false(self, mock_init):
        ml = MatchaLogger.__new__(MatchaLogger)
        ml._store = False
        assert ml.store is False

    @patch("matcha.utils.logging.MLFlowLogger.__init__", return_value=None)
    def test_store_setter_accepts_bool(self, mock_init):
        ml = MatchaLogger.__new__(MatchaLogger)
        ml._store = False
        ml.store = True
        assert ml.store is True

    @patch("matcha.utils.logging.MLFlowLogger.__init__", return_value=None)
    def test_store_setter_rejects_non_bool(self, mock_init):
        ml = MatchaLogger.__new__(MatchaLogger)
        ml._store = False
        with pytest.raises(ValueError, match="must be set as bool"):
            ml.store = "yes"
