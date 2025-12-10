# modeling_my_custom_policy.py
import torch
import torch.nn as nn
from typing import Dict, Any

from lerobot.policies.pretrained import PreTrainedPolicy
from .configuration_audiosmolvla import AudioSmolVLAConfig

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class AudioSmolVLAPolicy(SmolVLAPolicy):
    config_class = AudioSmolVLAConfig
    name = "audiosmolvla"

    def __init__(
        self, config: AudioSmolVLAConfig, dataset_stats: Dict[str, Any] = None
    ):
        super().__init__(config, dataset_stats)
        ...
