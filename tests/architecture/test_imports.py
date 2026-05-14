"""Architecture guard tests — enforce layer boundaries and interface contracts."""

import ast
from pathlib import Path

_SRC_DIR: Path = Path(__file__).resolve().parent.parent.parent / "src" / "memory_bridge"


def _imports_of(module_path: Path) -> set[str]:
    """Extract all import target module paths from a Python source file."""
    targets: set[str] = set()
    source: str = module_path.read_text(encoding="utf-8")
    tree: ast.AST = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                targets.add(node.module)
    return targets


def _python_files(subdir: str) -> list[Path]:
    return sorted(p for p in (_SRC_DIR / subdir).rglob("*.py") if p.name != "__init__.py")


class TestLayerBoundaries:
    """core/ must not import from api/ or providers/."""

    def test_core_does_not_import_api(self) -> None:
        for f in _python_files("core"):
            imports: set[str] = _imports_of(f)
            api_imports: list[str] = sorted(
                i for i in imports if i.startswith("memory_bridge.api")
            )
            assert not api_imports, (
                f"{f.name} imports from api/: {api_imports}"
            )

    def test_core_does_not_import_providers(self) -> None:
        for f in _python_files("core"):
            imports: set[str] = _imports_of(f)
            provider_imports: list[str] = sorted(
                i for i in imports if i.startswith("memory_bridge.providers")
            )
            assert not provider_imports, (
                f"{f.name} imports from providers/: {provider_imports}"
            )

    def test_models_does_not_import_upper_layers(self) -> None:
        for f in _python_files("models"):
            imports: set[str] = _imports_of(f)
            bad: list[str] = sorted(
                i for i in imports
                if i.startswith("memory_bridge.api")
                or i.startswith("memory_bridge.providers")
                or i.startswith("memory_bridge.core")
                or i.startswith("memory_bridge.config")
            )
            assert not bad, (
                f"{f.name} imports from upper layers: {bad}"
            )

    def test_providers_does_not_import_api(self) -> None:
        for f in _python_files("providers"):
            imports: set[str] = _imports_of(f)
            api_imports: list[str] = sorted(
                i for i in imports if i.startswith("memory_bridge.api")
            )
            assert not api_imports, (
                f"{f.name} imports from api/: {api_imports}"
            )


class TestInterfaceContracts:

    def test_abstract_provider_has_required_methods(self) -> None:
        from memory_bridge.providers.base import AbstractLLMProvider
        assert hasattr(AbstractLLMProvider, "chat")
        assert hasattr(AbstractLLMProvider, "chat_stream")
        assert hasattr(AbstractLLMProvider, "close")


class TestProviderRegistryIsolation:

    def test_core_does_not_import_provider_registry(self) -> None:
        for f in _python_files("core"):
            imports: set[str] = _imports_of(f)
            registry_imports: list[str] = sorted(
                i for i in imports if "ProviderRegistry" in i or "registry" in i
            )
            assert not registry_imports, (
                f"{f.name} should not depend on ProviderRegistry: {registry_imports}"
            )
