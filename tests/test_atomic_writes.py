"""Tests for atomic write utility — correctness and crash resistance."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from unittest.mock import patch

import pytest


class TestAtomicWrite:
    """Tests for habitus.utils.atomic_write helper."""

    def test_writes_json_to_path(self, tmp_path: Path):
        """atomic_write creates the destination file with correct JSON."""
        from habitus.habitus.utils import atomic_write
        dest = str(tmp_path / "output.json")
        data = {"hello": "world", "n": 42}
        atomic_write(dest, data)
        assert Path(dest).exists()
        loaded = json.loads(Path(dest).read_text())
        assert loaded["hello"] == "world"
        assert loaded["n"] == 42

    def test_creates_parent_directory(self, tmp_path: Path):
        """atomic_write creates parent dirs if they don't exist."""
        from habitus.habitus.utils import atomic_write
        dest = str(tmp_path / "subdir" / "nested" / "data.json")
        atomic_write(dest, {"ok": True})
        assert Path(dest).exists()

    def test_no_leftover_tmp_file_on_success(self, tmp_path: Path):
        """On success, no .tmp file is left behind."""
        from habitus.habitus.utils import atomic_write
        dest = str(tmp_path / "clean.json")
        atomic_write(dest, {"x": 1})
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Unexpected .tmp files: {tmp_files}"

    def test_atomic_replace_no_partial_read(self, tmp_path: Path):
        """A concurrent reader should never see partial JSON (atomic replace)."""
        from habitus.habitus.utils import atomic_write
        dest = tmp_path / "concurrent.json"
        # Write initial data
        atomic_write(str(dest), {"v": 0})

        errors = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    text = dest.read_text()
                    if text:
                        json.loads(text)  # should always be valid JSON
                except json.JSONDecodeError as e:
                    errors.append(str(e))
                except FileNotFoundError:
                    pass  # brief moment during replace — acceptable

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        for i in range(50):
            atomic_write(str(dest), {"v": i, "data": "x" * 100})

        stop.set()
        t.join(timeout=2)
        assert errors == [], f"Partial JSON reads detected: {errors}"

    def test_no_tmp_file_left_on_error(self, tmp_path: Path):
        """When write fails (e.g. OS error), no .tmp file is left."""
        from habitus.habitus.utils import atomic_write
        from unittest.mock import patch
        import json

        dest = str(tmp_path / "fail.json")

        # Simulate a write failure by making os.fdopen raise
        with patch("os.fdopen", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                atomic_write(dest, {"key": "value"})

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Stray .tmp file: {tmp_files}"

    def test_overwrites_existing_file(self, tmp_path: Path):
        """atomic_write overwrites existing files correctly."""
        from habitus.habitus.utils import atomic_write
        dest = str(tmp_path / "overwrite.json")
        atomic_write(dest, {"a": 1})
        atomic_write(dest, {"b": 2})
        loaded = json.loads(Path(dest).read_text())
        assert loaded == {"b": 2}

    def test_handles_unicode_data(self, tmp_path: Path):
        """atomic_write handles unicode strings."""
        from habitus.habitus.utils import atomic_write
        dest = str(tmp_path / "unicode.json")
        data = {"greeting": "Héllo wörld 🏠"}
        atomic_write(dest, data)
        loaded = json.loads(Path(dest).read_text(encoding="utf-8"))
        assert loaded["greeting"] == "Héllo wörld 🏠"
