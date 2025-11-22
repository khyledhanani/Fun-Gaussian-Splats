from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class GaussianSplatMethod(ABC):
	"""
	Base interface all methods must implement.
	Encapsulates training loop hooks and checkpoint IO.
	"""

	def __init__(self, config: Dict[str, Any]) -> None:
		self.config = config

	@abstractmethod
	def train_step(self, batch: Dict[str, Any], iter_pct: float = 0.0) -> Dict[str, Any]:
		"""
		Run one optimization step and return logs (e.g., loss).
		"""

	@abstractmethod
	def save(self, path: Path) -> None:
		"""
		Save method state to path.
		"""

	@abstractmethod
	def load(self, path: Path, strict: bool = True) -> None:
		"""
		Load method state from path.
		"""


