from __future__ import annotations

import logging
from typing import Optional


def create_logger(name: Optional[str] = None) -> logging.Logger:
	logger = logging.getLogger(name if name is not None else "gsplat")
	if not logger.handlers:
		handler = logging.StreamHandler()
		formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
		handler.setFormatter(formatter)
		logger.addHandler(handler)
	logger.setLevel(logging.INFO)
	return logger


