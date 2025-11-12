from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List


class NerfDataset:
	"""
	Minimal placeholder dataset. Replace with LLFF / NeRF-Synthetic loaders.
	"""

	def __init__(self, root: str | Path, split: str = "train") -> None:
		self.root = Path(root)
		self.split = split
		self.samples: List[Dict[str, Any]] = []
		# TODO: populate from transforms.json + images

	def __len__(self) -> int:
		return len(self.samples)

	def __iter__(self) -> Iterator[Dict[str, Any]]:
		for sample in self.samples:
			yield sample


