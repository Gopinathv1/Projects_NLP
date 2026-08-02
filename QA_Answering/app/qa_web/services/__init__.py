"""Service layer: everything that touches the model lives here, not in routes."""
from .qa_service import QAService

__all__ = ["QAService"]
