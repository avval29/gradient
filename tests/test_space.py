import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_lite_space_contents(tmp_path):
    import make_space

    out = make_space.build_space(tmp_path / "s", gradio_version="6.32.0")
    names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert {"app.py", "README.md", "requirements.txt", "gradient/qa.py", "data/sample/photosynthesis.txt"} <= names
    assert not any(n.startswith(("tests", "scripts", "data/private", "models", "docs")) for n in names)
    assert not any("__pycache__" in n for n in names)
    readme = (out / "README.md").read_text()
    assert readme.startswith("---") and "sdk: gradio" in readme and "sdk_version: 6.32.0" in readme
    reqs = (out / "requirements.txt").read_text()
    assert "torch" not in reqs and "pytest" not in reqs and "gradio" not in reqs
    assert 'GRADIENT_BACKEND' not in (out / "app.py").read_text().split("from gradient")[0]


def test_full_space_sets_adapter_and_cpu_torch(tmp_path):
    import make_space

    out = make_space.build_space(tmp_path / "s", adapter="me/gradient-qg-lora", gradio_version="6.32.0")
    head = (out / "app.py").read_text().split("from gradient")[0]
    assert 'GRADIENT_BACKEND", "hf"' in head and 'GRADIENT_ADAPTER", "me/gradient-qg-lora"' in head
    reqs = (out / "requirements.txt").read_text()
    assert "peft" in reqs and "download.pytorch.org/whl/cpu" in reqs and "torch" in reqs


def test_built_space_is_self_contained_and_in_demo_mode(tmp_path):
    """Import the built app from inside the Space folder only: no module may come from the main repo."""
    import make_space

    out = make_space.build_space(tmp_path / "s", gradio_version="6.32.0")
    code = ("import os, app, gradient; print(gradient.__file__); print(os.environ['GRADIENT_DEMO']); "
            "print(app.DEMO, app.thumbs_up('quiz (Q&A)', 't', 'draft', '', ''))")
    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
    res = subprocess.run([sys.executable, "-c", code], cwd=out, capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stderr
    lines = res.stdout.strip().splitlines()
    assert str(out) in lines[0] and lines[1] == "1"
    assert "True" in lines[2] and "not saved" in lines[2]


def test_demo_mode_saves_nothing(tmp_path, monkeypatch):
    import app

    monkeypatch.setattr(app, "DEMO", True)
    monkeypatch.setattr(app, "LOG_PATH", str(tmp_path / "grades.jsonl"))
    monkeypatch.setattr(app, "FEEDBACK_LOG", str(tmp_path / "fb.jsonl"))
    table, score, fb, state = app.draft_grade("q", app.EXAMPLE_RUBRIC, "Plants use light, water and CO2.")
    msg = app.approve("Ms Rao", score, fb, state)
    assert "Demo mode: nothing was saved" in msg
    assert "not saved" in app.thumbs_down("quiz (Q&A)", "t", "draft", "", "bad")
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(app, "DEMO", False)
    assert app.approve("Ms Rao", score, fb, state).startswith("Saved:")
    assert (tmp_path / "grades.jsonl").exists()


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create", repo_id, kw.get("repo_type")))

    def upload_folder(self, folder_path, repo_id, repo_type):
        self.calls.append(("upload", repo_id, repo_type, sorted(p.name for p in Path(folder_path).iterdir())))


def test_publish_uploads_adapter_with_card_then_space(tmp_path):
    import publish_hf

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    api = FakeApi()
    url, adapter_id = publish_hf.publish("me", adapter_dir=adapter, api=api, workdir=tmp_path / "w")
    assert adapter_id == "me/gradient-qg-lora" and url == "https://huggingface.co/spaces/me/gradient"
    kinds = [(c[0], c[1], c[2]) for c in api.calls]
    assert kinds == [("create", "me/gradient-qg-lora", "model"), ("upload", "me/gradient-qg-lora", "model"),
                     ("create", "me/gradient", "space"), ("upload", "me/gradient", "space")]
    assert api.calls[1][3] == ["README.md", "adapter_config.json"]
    assert "app.py" in api.calls[3][3] and "requirements.txt" in api.calls[3][3]
    assert 'GRADIENT_ADAPTER", "me/gradient-qg-lora"' in (tmp_path / "w/space/app.py").read_text()


def test_publish_without_adapter_makes_lite_space(tmp_path):
    import publish_hf

    api = FakeApi()
    url, adapter_id = publish_hf.publish("me", adapter_dir=tmp_path / "missing", api=api, workdir=tmp_path / "w")
    assert adapter_id is None and [c[0] for c in api.calls] == ["create", "upload"]
    assert "GRADIENT_BACKEND" not in (tmp_path / "w/space/app.py").read_text().split("from gradient")[0]
