from typing import Dict, Type


class _Registry:
	"""
	Lightweight string-to-class registry for selecting method implementations at runtime.
	"""

	def __init__(self) -> None:
		self._name_to_type: Dict[str, Type] = {}

	def register(self, name: str):
		def decorator(cls: Type):
			lower = name.lower()
			if lower in self._name_to_type:
				raise ValueError(f"Method '{name}' is already registered")
			self._name_to_type[lower] = cls
			return cls

		return decorator

	def get(self, name: str) -> Type:
		lower = name.lower()
		if lower not in self._name_to_type:
			known = ", ".join(sorted(self._name_to_type))
			raise KeyError(f"Unknown method '{name}'. Known: {known}")
		return self._name_to_type[lower]


METHOD_REGISTRY = _Registry()


