from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_play_youtube_module():
    path = Path(__file__).resolve().parents[1] / "skills" / "utilities" / "tools.py"
    spec = importlib.util.spec_from_file_location("play_youtube_test", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_play_youtube_resolves_first_video_url(mocker):
    module = _load_play_youtube_module()

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"videoRenderer":{"videoId":"dQw4w9WgXcQ"}}'

    mocker.patch.object(module.urllib.request, "urlopen", return_value=_Resp())
    url, video_id, resolved = module._resolve_first_video_url("https://www.youtube.com/results?search_query=test")
    assert resolved is True
    assert video_id == "dQw4w9WgXcQ"
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&autoplay=1"


def test_play_youtube_falls_back_to_search_url_on_fetch_failure(mocker):
    module = _load_play_youtube_module()
    search_url = "https://www.youtube.com/results?search_query=test"
    mocker.patch.object(module.urllib.request, "urlopen", side_effect=RuntimeError("boom"))
    url, video_id, resolved = module._resolve_first_video_url(search_url)
    assert resolved is False
    assert video_id == ""
    assert url == search_url
