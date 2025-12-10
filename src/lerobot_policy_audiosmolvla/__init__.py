# __init__.py
"""AudioSmolVLA policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_audiosmolvla import AudioSmolVLAConfig
from .modeling_audiosmolvla import AudioSmolVLAPolicy
from .processor_audiosmolvla import make_my_custom_policy_pre_post_processors

__all__ = [
    "AudioSmolVLAConfig",
    "AudioSmolVLAPolicy",
    "make_my_custom_policy_pre_post_processors",
]
