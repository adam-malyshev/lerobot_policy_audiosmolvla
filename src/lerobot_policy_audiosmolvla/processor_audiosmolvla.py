import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass

from .configuration_audiosmolvla import AudioSmolVLAConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
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
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from torchvision.transforms import Compose, Lambda

from .dymn import AugmentMelSTFT


def make_audiosmolvla_pre_post_processors(
    config: AudioSmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
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
@ProcessorStepRegistry.register(name="custom_audio_processor")
class CustomAudioProcessorStep(AudioProcessorStep):
    input_audio_chunk_duration: float = 1.0
    input_sample_rate: int = 44100
    intermediate_sample_rate: int = 32000
    n_fft: int = 1024
    hop_length: int = 320
    n_mels: int = 128

    def __post_init__(self):
        self.preprocessor = AugmentMelSTFT(
            n_mels=self.n_mels,
            sr=self.intermediate_sample_rate,
            win_length=800,  # Fixed from DyMN default
            hopsize=self.hop_length,
            n_fft=self.n_fft,
        )

        self.mel_spectrogram_transform = Compose(
            [
                Lambda(
                    lambda x: x.mean(dim=1) if x.ndim == 3 else x
                ),  # Force mono if not handled
                torchaudio.transforms.Resample(
                    orig_freq=self.input_sample_rate,
                    new_freq=self.intermediate_sample_rate,
                ),
                self.preprocessor,
                Lambda(lambda x: x.unsqueeze(1)),
            ]
        )
