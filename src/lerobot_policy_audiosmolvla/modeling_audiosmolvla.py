# modeling_my_custom_policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, List
import math

from lerobot.policies.pretrained import PreTrainedPolicy
from .configuration_audiosmolvla import AudioSmolVLAConfig

from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    VLAFlowMatching,
    SmolVLMWithExpertModel,
    pad_tensor,
    make_att_2d_masks,
    resize_with_pad,
    pad_vector,
    aloha_gripper_to_angular,
    aloha_gripper_from_angular,
    aloha_gripper_from_angular_inv
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    OBS_IMAGES
)
from lerobot.policies.utils import populate_queues

from transformers import AddedToken
from .dymn import DyMNMedium


class AudioProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class AudioSmolVLAPolicy(SmolVLAPolicy):
    config_class = AudioSmolVLAConfig
    name = "audiosmolvla"

    def __init__(
        self, config: AudioSmolVLAConfig, dataset_stats: Dict[str, Any] = None
    ):
        PreTrainedPolicy.__init__(self, config, dataset_stats)
        config.validate_features()
        self.config = config

        self.model = AudioVLAFlowMatching(config)
        
        self.reset()


    def prepare_audio(self, batch):
        """Convert audio to """
        audios = []
        audio_masks = []
        present_audio_keys = [key for key in self.config.audio_features if key in batch]
        missing_audio_keys = [key for key in self.config.audio_features if key not in batch]

        if len(present_audio_keys) == 0:
            raise ValueError(
                f"All audio features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.audio_features})"
            )
        # Preprocess audio features present in the batch
        for key in present_audio_keys:
            audio = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]

            bsize = audio.shape[0]
            device = audio.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            audios.append(audio)
            audio_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded audios.
        for num_empty_mics in range(len(missing_audio_keys)):
            if num_empty_mics >= self.config.empty_mics:
                break
            audio = torch.ones_like(audio) * -1
            mask = torch.zeros_like(mask)
            audios.append(audio)
            audio_masks.append(mask)
        return audios, audio_masks

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        
        audio_list, audio_masks_list = self.prepare_audio(batch)

        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        
        losses = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, 
            audio_list, audio_masks_list, 
            state, actions, noise, time
        )
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # For backward pass
        loss = losses.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        
        audio_list, audio_masks_list = self.prepare_audio(batch)

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, 
            audio_list, audio_masks_list, 
            state, noise=noise
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions


class AudioVLAFlowMatching(VLAFlowMatching):
    def __init__(self, config: AudioSmolVLAConfig):
        super().__init__(config)

        tokenizer = self.vlm_with_expert.processor.tokenizer
        vlm_model = self.vlm_with_expert.vlm

        self.audio_start_token_str = AddedToken("<|audio_start|>", normalized=False, special=True, lstrip=False, rstrip=False)
        self.audio_end_token_str = AddedToken("<|audio_end|>", normalized=False, special=True, lstrip=False, rstrip=False)
        
        audio_tokens = [self.audio_start_token_str, self.audio_end_token_str]
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": audio_tokens})

        if num_added > 0:
            vlm_model.resize_token_embeddings(len(tokenizer))
            with torch.no_grad():
                audio_ref_ids = tokenizer("audio", add_special_tokens=False).input_ids
                if len(audio_ref_ids) > 0:
                    audio_ref_id = audio_ref_ids[0]
                    ref_embedding = vlm_model.get_input_embeddings().weight[audio_ref_id]
                    embedding_layer = vlm_model.get_input_embeddings()
                    embedding_layer.weight[-num_added:] = ref_embedding.clone()

        self.audio_start_token_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_end_token_id = tokenizer.convert_tokens_to_ids("<|audio_end|>")

        self.audio_input_dim = 960 
        self.lm_hidden_size = self.vlm_with_expert.config.text_config.hidden_size
        self.audio_projector = AudioProjector(self.audio_input_dim, self.lm_hidden_size)
        
        self.audio_encoder = DyMNMedium(pretrained=True, device=config.device, target_embed_dim=None)
        self.audio_encoder.device

    def embed_audio(self, audio_spectrogram: torch.Tensor):
        x = self.audio_encoder(audio_spectrogram)
        audio_hidden_states = self.audio_projector(x)
        return audio_hidden_states

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        audio_list,
        audio_masks_list,
        state: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images, audio, and language tokens to prepare for SmolVLM transformer processing."""
        embs = []
        pad_masks = []
        att_masks = []
        
        for _img_idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * (num_img_embs)

            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])

        if audio_list is not None and len(audio_list) > 0:
            for audio, _ in zip(audio_list, audio_masks_list):
                audio_emb = self.embed_audio(audio)

                bsize = audio_emb.shape[0]
                device = audio_emb.device

                start_emb = self.vlm_with_expert.embed_language_tokens(
                    torch.tensor([self.audio_start_token_id], device=device)
                ).expand(bsize, -1, -1)
                
                end_emb = self.vlm_with_expert.embed_language_tokens(
                    torch.tensor([self.audio_end_token_id], device=device)
                ).expand(bsize, -1, -1)

                audio_emb = torch.cat([start_emb, audio_emb, end_emb], dim=1)

                audio_emb = audio_emb * math.sqrt(audio_emb.shape[-1])

                embs.append(audio_emb)

                feat_len = audio_emb.shape[1]
                full_audio_mask = torch.ones(bsize, feat_len, dtype=torch.bool, device=device)
                pad_masks.append(full_audio_mask)

                att_masks += [0] * feat_len

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        
        bsize = state_emb.shape[0]
        device = state_emb.device
        states_seq_len = state_emb.shape[1]
        
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1] * (states_seq_len) # State attends to prefix

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, audio_list, audio_masks_list, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, audio_list, audio_masks_list, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, audio_list, audio_masks_list, state, noise=None) -> Tensor:
        """Do a full inference forward and compute the action."""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, audio_list, audio_masks_list, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t += dt * v_t
            time += dt
        return x_t
