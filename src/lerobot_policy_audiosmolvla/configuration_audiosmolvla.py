# configuration_my_custom_policy.py
from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode, FeatureType

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel


@PreTrainedConfig.register_subclass("audiosmolvla")
@dataclass
class AudioSmolVLAConfig(SmolVLAConfig):
    """Configuration class for MyCustomPolicy.

    Args:
        n_obs_steps: Number of observation steps to use as input
        horizon: Action prediction horizon
        n_action_steps: Number of action steps to execute
        hidden_dim: Hidden dimension for the policy network

        # policy specific args:
        audio_model_name: The name/path of the DyMN model to load (e.g. 'dymn10_as').
        audio_projector_repo_id: HF Hub ID where the projector weights are stored.
        audio_token_dim: The hidden dimension of the audio tokens (must match projector output).
        target_audio_len_samples: The target length in samples for the audio encoder (10s * 32kHz).
        sampling_rate: The target sampling rate for the audio encoder.
    """
    audio_model_name: str = "dymn10_as" 
    audio_projector_repo_id: str = "shivamg05/SmolVLA-Audio-Projector"
    audio_token_dim: int = 960
    
    target_audio_len_samples: int = 320000 
    sampling_rate: int = 32000

    def __post_init__(self):
        super().__post_init__()
        if self.audio_token_dim <= 0:
            raise ValueError(f"audio_token_dim must be positive, got {self.audio_token_dim}")

    def validate_features(self) -> None:
        """Validate input/output feature compatibility."""
        # Implement validation logic for your policy's requirements
        
        # Run parent validation
        super().validate_features()

        # check for audio features
        # scan input_features to make sure at least one has type AUDIO or is named 'audio'
        has_audio = False
        
        for key in self.input_features:
            if "audio" in key: # requires for the dataset used @ training time to have a key like observation.audio
                has_audio = True
                break
        
        if not has_audio:
            # We don't raise an error immediately to allow for 'vision-only' fallback 
            # if that's desired, but for this specific policy, we warn or raise.
            print("WARNING: No Audio feature detected in input_features. "
                  "AudioSmolVLA expects an 'observation.audio' key.")

