# routes __init__ - register all routers
from backend.routes import health, applicants, underwriting, policy, model_info, agent, dashboard

__all__ = ["health", "applicants", "underwriting", "policy", "model_info", "agent", "dashboard"]
