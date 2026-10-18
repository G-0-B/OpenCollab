"""Public single-agent profile names.

Base currently selects Single2; ``resolve_profile_name`` returns that implementation.
"""

from opencollab.bootstrap.agent_profiles import BASE_PROFILE, resolve_profile_name

__all__ = ["BASE_PROFILE", "resolve_profile_name"]
