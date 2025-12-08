# Re-export page modules so `from pages_logic import ...` works
from . import home, algorithms, run_models, publications, contact, chat_with_agent

__all__ = [
    "home",
    "algorithms",
    "run_models",
    "publications",
    "contact",
    "chat_with_agent",
]
