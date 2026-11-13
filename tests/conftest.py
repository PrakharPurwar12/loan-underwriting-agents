import dotenv
import pytest


dotenv.load_dotenv = lambda *args, **kwargs: False


@pytest.fixture(autouse=True)
def isolate_groq_environment(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
