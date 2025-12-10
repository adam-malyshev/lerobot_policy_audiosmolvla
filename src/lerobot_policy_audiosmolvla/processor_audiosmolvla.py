# processor_my_custom_policy.py
from typing import Dict, Any
import torch

from dataclasses import dataclass, field
from .configuration_audiosmolvla import AudioSmolVLAConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    AudioProcessorStep,
)
from lerobot.policies.smolvla.processor_smolvla import SmolVLANewLineProcessor

from lerobot.datasets.utils import DEFAULT_AUDIO_CHUNK_DURATION
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from torchaudio.transforms import MelSpectrogram, Resample
from torchvision.transforms import Compose, Lambda, Resize

from lerobot.datasets.utils import DEFAULT_AUDIO_CHUNK_DURATION
from lerobot.utils.constants import OBS_AUDIO

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


def make_audiosmolvla_pre_post_processors(
    config: AudioSmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_steps = [
        RenameObservationsProcessorStep(
            rename_map={}
        ),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        SmolVLANewLineProcessor(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        # From EfficientAT AugmentMelSTFT specs https://github.com/fschmid56/EfficientAT/tree/main
        CustomAudioProcessorStep(),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


@dataclass
@ProcessorStepRegistry.register(name="audio_processor")
class CustomAudioProcessorStep(AudioProcessorStep):
    input_audio_chunk_duration: float = 1.0
    input_sample_rate: int = 44100
    intermediate_sample_rate: int = 32000
    n_fft: int = 1024
    hop_length: int = 320
    n_mels: int = 128

    def __post_init__(self):
        self.mel_spectrogram_transform = Compose(
            [
                Lambda(
                    lambda x: x.mean(dim=1)
                ),  # Average over all channels (second dimension after batch)
                Resample(
                    orig_freq=self.input_sample_rate,
                    new_freq=self.intermediate_sample_rate,
                ),
                MelSpectrogram(
                    sample_rate=self.intermediate_sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                    power=2,  # Power spectrum
                ),
                Lambda(lambda x: (x + 1e-5).log()),
                Lambda(lambda x: (x + 4.5) / 5.0),
                # No resizing for DyMN
                # Resize(
                #     (self.output_height, self.output_width)
                # ),  # Resize spectrogram to output_height×output_width
                Lambda(lambda x: x.unsqueeze(0)),
            ]
        )
