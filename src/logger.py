# -*- coding: utf-8 -*-
"""Lightweight print-based logger for operational pipelines.

Uses Python's built-in print for console output and mirrors messages to a
UTF-8 log file. The standard logging module is intentionally avoided to
follow the project log strategy.

Founded in 2026-07-16
Modified in 2026-07-26
@author: yinlb
"""

import pathlib
import typing

import arrow


# Ordered log levels from least to most restrictive
_LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR')


class Logger:
    """Print-based logger that also writes to a run-specific log file.

    Attributes:
        _log_dir: Directory containing log files.
        _log_level: Minimum log level to emit.
        _run_id: Identifier for this run, used in the log filename.
        _file_path: Full path to the log file.
        _file: Open log file handle.
    """

    def __init__(
        self,
        log_dir: str,
        log_level: str = 'INFO',
        run_id: typing.Optional[str] = None
    ) -> None:
        """
        Initialize logger with log directory, level, and run id.

        Args:
            log_dir: Directory where log files are written.
            log_level: Minimum level to emit (DEBUG/INFO/WARNING/ERROR).
            run_id: Optional run identifier; defaults to current timestamp.
        """
        self._log_dir = pathlib.Path(log_dir)
        self._log_level = log_level.upper()
        if self._log_level not in _LEVELS:
            raise ValueError(
                f'Invalid log level: {log_level}. '
                f'Choose from {_LEVELS}.'
            )

        if run_id is None:
            run_id = arrow.now().format('YYYYMMDD_HHmmss')
        self._run_id = run_id

        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = self._log_dir / f'ops_{self._run_id}.log'
        self._file = open(self._file_path, 'a', encoding='utf-8')

        self.info(f'Logger initialized: {self._file_path}')

    def debug(self, message: str) -> None:
        """Log a DEBUG message.

        Args:
            message: Log message text.
        """
        self._write('DEBUG', message)

    def info(self, message: str) -> None:
        """Log an INFO message.

        Args:
            message: Log message text.
        """
        self._write('INFO', message)

    def warning(self, message: str) -> None:
        """Log a WARNING message.

        Args:
            message: Log message text.
        """
        self._write('WARNING', message)

    def error(self, message: str) -> None:
        """Log an ERROR message.

        Args:
            message: Log message text.
        """
        self._write('ERROR', message)

    def close(self) -> None:
        """Close the log file handle."""
        if self._file and not self._file.closed:
            self._file.close()

    def copy(self) -> 'Logger':
        """Return a new Logger with the same configuration.

        The new instance appends to the same log file (same run_id).

        Returns:
            Logger: A new instance sharing this instance's settings.
        """
        return Logger(
            log_dir=str(self._log_dir),
            log_level=self._log_level,
            run_id=self._run_id
        )

    def _should_log(self, level: str) -> bool:
        """Return True if level should be emitted given current threshold.

        Args:
            level: Candidate log level string.

        Returns:
            True if level is at or above the configured threshold.
        """
        return _LEVELS.index(level) >= _LEVELS.index(self._log_level)

    def _format_message(self, level: str, message: str) -> str:
        """Format message with timestamp and level.

        Args:
            level: Log level string.
            message: Log message text.

        Returns:
            Formatted log line with timestamp prefix.
        """
        timestamp = arrow.now().format('YYYY-MM-DD HH:mm:ss')
        return f'{timestamp} [{level}] {message}'

    def _write(self, level: str, message: str) -> None:
        """Print to console and append to log file if level is enabled.

        Args:
            level: Log level string.
            message: Log message text.
        """
        if not self._should_log(level):
            return
        formatted = self._format_message(level, message)
        print(formatted)
        self._file.write(formatted + '\n')
        self._file.flush()


# Module-level default logger instance
_default_logger: typing.Optional[Logger] = None


def setup_logger(
    log_dir: str,
    log_level: str = 'INFO',
    run_id: typing.Optional[str] = None
) -> Logger:
    """
    Create and store the default logger instance.

    Args:
        log_dir: Directory where log files are written.
        log_level: Minimum level to emit.
        run_id: Optional run identifier.

    Returns:
        Logger: The configured default logger.
    """
    global _default_logger
    _default_logger = Logger(
        log_dir=log_dir,
        log_level=log_level,
        run_id=run_id
    )
    return _default_logger


def get_logger() -> Logger:
    """
    Return the default logger instance.

    Returns:
        Logger: The configured default logger.

    Raises:
        RuntimeError: If setup_logger has not been called.
    """
    if _default_logger is None:
        raise RuntimeError(
            'Logger not configured. Call setup_logger() first.'
        )
    return _default_logger


def debug(message: str) -> None:
    """Delegate DEBUG message to default logger.

    Args:
        message: Log message text.
    """
    get_logger().debug(message)


def info(message: str) -> None:
    """Delegate INFO message to default logger.

    Args:
        message: Log message text.
    """
    get_logger().info(message)


def warning(message: str) -> None:
    """Delegate WARNING message to default logger.

    Args:
        message: Log message text.
    """
    get_logger().warning(message)


def error(message: str) -> None:
    """Delegate ERROR message to default logger.

    Args:
        message: Log message text.
    """
    get_logger().error(message)
