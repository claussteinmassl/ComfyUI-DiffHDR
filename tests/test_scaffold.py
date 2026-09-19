import pytest

import diffhdr
from tests import conftest


def test_version():
    assert diffhdr.__version__ == "0.1.0"


@pytest.mark.parametrize("value, expected", [
    (None, False), ("", False), ("0", False), ("false", False), ("False", False),
    ("1", True), ("true", True), ("yes", True),
])
def test_require_comfy_env_flag(monkeypatch, value, expected):
    """``DIFFHDR_REQUIRE_COMFY`` turns the ComfyUI skip into a hard error in CI."""
    monkeypatch.delenv("DIFFHDR_REQUIRE_COMFY", raising=False)
    if value is not None:
        monkeypatch.setenv("DIFFHDR_REQUIRE_COMFY", value)
    assert conftest.require_comfy() is expected


def test_require_comfy_without_comfy_is_a_usage_error(monkeypatch):
    monkeypatch.setenv("DIFFHDR_REQUIRE_COMFY", "1")
    monkeypatch.setattr(conftest, "_has_comfy", lambda: False)
    with pytest.raises(pytest.UsageError, match="DIFFHDR_REQUIRE_COMFY"):
        conftest.pytest_configure(None)


def test_reference_skip_reason_names_the_clone_command():
    assert "git clone https://github.com/Eyeline-Labs/DiffHDR .dev/reference/DiffHDR" in conftest.REFERENCE_HINT
