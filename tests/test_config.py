import pytest

from gradient.config import make_backend, make_retriever
from gradient.generator import EchoBackend
from gradient.rag import Retriever


def test_defaults_are_lightweight():
    assert isinstance(make_backend(), EchoBackend)
    assert isinstance(make_retriever(data_dir="data/sample"), Retriever)


def test_unknown_options_fail_loudly():
    with pytest.raises(ValueError):
        make_backend("nope")
    with pytest.raises(ValueError):
        make_retriever("nope", "data/sample")
