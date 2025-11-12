from gsplat.registry import METHOD_REGISTRY


def test_registry_has_methods():
	Original = METHOD_REGISTRY.get("original")
	VQ = METHOD_REGISTRY.get("vq")
	assert Original is not None
	assert VQ is not None


