from dataclasses import dataclass, field
import logging
import time
from typing import TypeAlias, cast
import os

from accelerate import load_checkpoint_and_dispatch

from einops import rearrange
from hydra.utils import instantiate
from peft import LoraConfig, get_peft_model
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from safetensors.torch import load_file
import json
from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)

WAN_HF_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
WAN22_HF_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"


def hf_download(filename: str, repo_id: str = WAN_HF_REPO_ID) -> str:
    """Download a file from the specified HuggingFace repo to HF cache."""
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return path


def ensure_file(path: str | None, hf_filename: str, repo_id: str = WAN_HF_REPO_ID) -> str:
    """Return a valid local path: use `path` if it exists, otherwise download from HuggingFace."""
    if path is not None and os.path.exists(path):
        return path
    return hf_download(hf_filename, repo_id)

from torch.distributions import Beta
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torchvision.transforms import v2
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from groot.vla.model.n1_5.action_head.base_action_head import ActionHead
from groot.vla.model.dreamzero.modules.flow_match_scheduler import FlowMatchScheduler
from groot.vla.model.dreamzero.modules.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from groot.vla.model.dreamzero.modules.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler


KVCacheType: TypeAlias = torch.Tensor

@dataclass
class WANPolicyHeadConfig(PretrainedConfig):
    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )
    tiled: bool = field(default=True, metadata={"help": "Whether to use tiled input."})
    tile_size_height: int = field(default=34, metadata={"help": "Tile size height."})
    tile_size_width: int = field(default=34, metadata={"help": "Tile size width."})
    tile_stride_height: int = field(default=18, metadata={"help": "Tile stride height."})
    tile_stride_width: int = field(default=16, metadata={"help": "Tile stride width."})
    num_frame_per_block: int = field(default=1, metadata={"help": "Number of frames per block."})
    # Target video (H, W) for Wan22 resize. When set, videos are resized to this before VAE so latent
    # spatial size matches. Use height/width divisible by 32 for WanVideoVAE38 (16x) so latent H,W are even.
    target_video_height: int | None = field(default=None, metadata={"help": "Target video height for resize (e.g. 160 for even latent with VAE38)."})
    target_video_width: int | None = field(default=None, metadata={"help": "Target video width for resize (e.g. 320)."})

    lora_rank: int = field(default=4, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=4, metadata={"help": "LoRA alpha."})
    lora_target_modules: str = field(default="q,k,v,o,ffn.0,ffn.2")
    init_lora_weights: str = field(default="kaiming", metadata={"help": "LoRA initialization method."})
    train_architecture: str= field(default="lora", metadata={"help": "Train architecture."})
    skip_component_loading: bool = field(default=False, metadata={"help": "Skip loading individual component weights (used when loading from full pretrained model)."})

    use_gradient_checkpointing: bool = field(default=True, metadata={"help": "Whether to use gradient checkpointing."})
    qformer_cfg: dict = field(default=None, metadata={"help": "Qformer configuration."})
    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    # High noise emphasis for BASE (coupled) training - applies Beta distribution to BOTH video and action together
    use_high_noise_emphasis: bool = field(
        default=False, metadata={"help": "Use Beta distribution for noise sampling (biases BOTH video and action towards high noise levels together)."}
    )
    high_noise_beta_alpha: float = field(
        default=3.0, metadata={"help": "Beta alpha for high noise emphasis. Beta(3,1): mean=0.75, Beta(5,1): mean=0.83. Higher = more high noise bias."}
    )
    # Decoupled noise sampling config for training-inference alignment
    # When enabled: video uses Beta(alpha,beta) biased towards high noise, action uses independent uniform
    decouple_video_action_noise: bool = field(
        default=False, metadata={"help": "Decouple video/action noise: video uses Beta distribution (high noise bias), action uses independent uniform."}
    )
    video_noise_beta_alpha: float = field(
        default=3.0, metadata={"help": "Beta alpha for video noise. Beta(3,1): mean=0.75, Beta(5,1): mean=0.83. Higher alpha = more bias to high noise."}
    )
    video_noise_beta_beta: float = field(
        default=1.0, metadata={"help": "Beta beta for video noise. Keep at 1.0."}
    )
    # Decoupled inference config - allows video to stay noisy while action fully denoises
    decouple_inference_noise: bool = field(
        default=False, metadata={"help": "Use decoupled noise schedules during inference (video stays noisy, action fully denoises)."}
    )
    video_inference_final_noise: float = field(
        default=0.8, metadata={"help": "Final noise level for video during decoupled inference (0.0-1.0). E.g., 0.8 means video ends at 80% noise."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)
    defer_lora_injection: bool = field(default=False, metadata={"help": "Defer LoRA injection until after loading pretrained weights."})

    vl_self_attention_cfg: dict = field(default=None)
    text_encoder_cfg: dict = field(default=None)
    image_encoder_cfg: dict = field(default=None)
    vae_cfg: dict = field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class WANPolicyHead(ActionHead):
    config_class = WANPolicyHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: WANPolicyHeadConfig,
    ):
        super().__init__()
        self.tiled = config.tiled
        self.tile_size_height = config.tile_size_height
        self.tile_size_width = config.tile_size_width
        self.tile_stride_height = config.tile_stride_height
        self.tile_stride_width = config.tile_stride_width
        self.num_frame_per_block = config.num_frame_per_block
        self.hidden_size = config.hidden_size
        self.num_frames = config.num_frames
        self.text_encoder = instantiate(config.text_encoder_cfg)
        self.image_encoder = instantiate(config.image_encoder_cfg)
        self.vae = instantiate(config.vae_cfg)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.model_names = ['text_encoder']

        self.num_inference_steps = 16 
        self.seed = 1140
        self.cfg_scale = 5.0
        self.denoising_strength = 1.0
        self.sigma_shift = 5.0
        self.kv_cache1: KVCacheType | None = None
        self.kv_cache_neg: KVCacheType | None = None
        self.crossattn_cache: KVCacheType | None = None
        self.crossattn_cache_neg: KVCacheType | None = None

        self.global_step = 0
        self.max_steps = 0
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_target_modules = config.lora_target_modules
        self.init_lora_weights = config.init_lora_weights
        self.train_architecture = config.train_architecture
        self.clip_feas = None
        self.ys = None
        self.current_start_frame = 0
        self.language = None

        self.ip_rank = 0
        self.ip_size = 1
        self.ip_group = None
        
        self._device = "cuda"
        self.dynamic_cache_schedule = os.getenv("DYNAMIC_CACHE_SCHEDULE", "False").lower() == "true"


        num_dit_steps = 8
        if os.getenv("NUM_DIT_STEPS") is not None:
            num_dit_steps = int(os.getenv("NUM_DIT_STEPS"))
        if num_dit_steps == 5:
            self.dit_step_mask = [True, True, True, False, False, False, False, True, False, False, False, False, True, False, False, False]
        elif num_dit_steps == 6:
            self.dit_step_mask = [True, True, False, False, False, True, False, False, False, False, True, False, False, False, True, True]
        elif num_dit_steps == 7:
            self.dit_step_mask = [True, True, True, False, False, False, True, False, False, False, True, False, False, False, True, True]
        elif num_dit_steps == 8:
            self.dit_step_mask = [True, True, True, False, False, False, True, False, False, False, True, False, False, True, True, True]
        else:
            self.dit_step_mask = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
        assert self.dit_step_mask[0] == True, "first step must be True"

        self.normalize_video = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        if self.training:
            self.scheduler.set_timesteps(1000, training=True)
        
        
        self.input_embedding_dim = config.input_embedding_dim

        self.cpu_offload = False

        self.model = instantiate(config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        
        text_enc_path = ensure_file(
            self.text_encoder.text_encoder_pretrained_path,
            "models_t5_umt5-xxl-enc-bf16.pth",
        )
        self.text_encoder.load_state_dict(torch.load(text_enc_path, map_location='cpu'))

        img_enc_path = ensure_file(
            self.image_encoder.image_encoder_pretrained_path,
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        )
        self.image_encoder.model.load_state_dict(torch.load(img_enc_path, map_location='cpu'), strict=False)

        # Wan2.2 (WanVideoVAE38, z_dim=48) uses Wan2.2_VAE.pth; Wan2.1 uses Wan2.1_VAE.pth
        vae_hf_filename = "Wan2.2_VAE.pth" if getattr(self.vae, "z_dim", 16) == 48 else "Wan2.1_VAE.pth"
        vae_repo_id = WAN22_HF_REPO_ID if getattr(self.vae, "z_dim", 16) == 48 else WAN_HF_REPO_ID
        vae_path = ensure_file(
            self.vae.vae_pretrained_path,
            vae_hf_filename,
            repo_id=vae_repo_id,
        )
        self.vae.model.load_state_dict(torch.load(vae_path, map_location='cpu'))

        if not config.skip_component_loading:
            dit_dir = self.model.diffusion_model_pretrained_path
            # Wan2.2 (in_dim=48) uses Wan2.2-TI2V-5B repo; Wan2.1 uses Wan2.1-I2V-14B-480P
            dit_repo_id = WAN22_HF_REPO_ID if getattr(self.model, "in_dim", 16) == 48 else WAN_HF_REPO_ID
            if dit_dir is None or not os.path.isdir(dit_dir):
                index_path = hf_hub_download(repo_id=dit_repo_id, filename="diffusion_pytorch_model.safetensors.index.json")
                dit_dir = os.path.dirname(index_path)
                with open(index_path, 'r') as f:
                    index = json.load(f)
                for shard_file in set(index["weight_map"].values()):
                    hf_hub_download(repo_id=dit_repo_id, filename=shard_file)

            if dit_dir is not None:
                safetensors_path = os.path.join(dit_dir, "diffusion_pytorch_model.safetensors")
                safetensors_index_path = os.path.join(dit_dir, "diffusion_pytorch_model.safetensors.index.json")
                state_dict = {}

                if os.path.exists(safetensors_index_path):
                    # Handle sharded safetensors
                    print(f"Loading sharded safetensors using index: {safetensors_index_path}")

                    with open(safetensors_index_path, 'r') as f:
                        index = json.load(f)

                    # Load each shard
                    for shard_file in set(index["weight_map"].values()):
                        shard_path = os.path.join(dit_dir, shard_file)
                        print(f"Loading shard: {shard_path}")
                        shard_state_dict = load_file(shard_path)
                        state_dict.update(shard_state_dict)

                elif os.path.exists(safetensors_path):
                    # Handle single safetensors file
                    print(f"Loading weights from safetensors: {safetensors_path}")
                    state_dict = load_file(safetensors_path)

                else:
                    raise ValueError(f"No safetensors file found at {safetensors_path} or {safetensors_index_path}")

                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

                if missing_keys:
                    print(f"Missing keys when loading pretrained weights: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")

                print("Successfully loaded pretrained weights")
        else:
            print("Skipping individual component loading (loading from full pretrained model)")
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        # Video noise Beta distribution (biased towards high noise levels when enabled)
        self.video_beta_dist = Beta(config.video_noise_beta_alpha, config.video_noise_beta_beta)
        # High noise emphasis Beta distribution for coupled training (applies to both video and action)
        self.high_noise_beta_dist = Beta(config.high_noise_beta_alpha, 1.0)
        # self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self._noise_logged = False
        self.defer_lora_injection = config.defer_lora_injection
        print("defer_lora_injection@@", self.defer_lora_injection)
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

        if self.train_architecture == "lora" and not self.defer_lora_injection:
            print("Adding LoRA to model")
            for p in self.parameters():
                p.requires_grad = False
            self.model = self.add_lora_to_model(
                self.model,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_target_modules=self.lora_target_modules,
                init_lora_weights=self.init_lora_weights,
            )
            self.model.state_encoder.requires_grad_(True)
            self.model.action_encoder.requires_grad_(True)
            self.model.action_decoder.requires_grad_(True)
        elif self.train_architecture == "lora" and self.defer_lora_injection:
            print("Deferring LoRA injection until after pretrained weights are loaded")
        else:
            self.print_trainable_params()

        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        if not self.defer_lora_injection:
            self.print_trainable_params()


    def print_trainable_params(self):
        """Print trainable parameters of the diffusion model."""
        trainable_params = []
        total_params = 0
        trainable_total = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params.append(name)
                trainable_total += param.numel()
                
        print(f"Total parameters in diffusion model: {total_params:,}")
        print(f"Trainable parameters in diffusion model: {trainable_total:,}")
        # print(trainable_params)


    def inject_lora_after_loading(self):
        """
        Inject LoRA adapters after pretrained weights have been loaded.
        This should be called when defer_lora_injection=True.
        """
        if self.train_architecture == "lora":
            print("Injecting LoRA after loading pretrained weights")
            for p in self.parameters():
                p.requires_grad = False
            self.model = self.add_lora_to_model(
                self.model,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_target_modules=self.lora_target_modules,
                init_lora_weights=self.init_lora_weights,
            )
            self.model.state_encoder.requires_grad_(True)
            self.model.action_encoder.requires_grad_(True)
            self.model.action_decoder.requires_grad_(True)
            # self.model.registers.requires_grad_(True)
            # self.model.time_modality_projection.requires_grad_(True)
            
            self.text_encoder.requires_grad_(False)
            self.image_encoder.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.print_trainable_params()
        else:
            print("LoRA injection not needed (train_architecture != 'lora')")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_diffusion_model:
                self.model.eval()
            self.text_encoder.eval()
            self.image_encoder.eval()
            self.vae.eval()
    
    
    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.dtype,
                computation_device='cuda',
            ),
        )

        self.cpu_offload = True

    def load_models_to_device(self, loadmodel_names=[]):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)
                if model is not None:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        print("offloadd")
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                # print("offload", module)
                                module.offload()
                    else:
                        print("tocpu")
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                    print("onload")
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            # print("onload", module)
                            module.onload()
                else:
                    print("togpu")
                    model.to(self._device)
        # fresh the cuda cache
        torch.cuda.empty_cache()

    def _create_kv_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        frame_seqlen: int,
    ) -> tuple[KVCacheType, KVCacheType]:
        """
        Initialize a Per-GPU KV cache for the Wan model.
        Use the model's num_heads and head_dim (5B has 24 heads, 14B has 40).
        """
        num_heads = self.model.num_heads
        head_dim = self.model.dim // num_heads
        kv_cache1: KVCacheType = []
        kv_cache_neg: KVCacheType = []
        for _ in range(self.model.num_layers):
            kv_cache1.append(
                torch.zeros([2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device),
            )
            kv_cache_neg.append(
                torch.zeros([2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device),
            )

        return kv_cache1, kv_cache_neg

    def _create_crossattn_caches(
        self, batch_size: int, dtype: torch.dtype, device: torch.device,
    ) -> tuple[KVCacheType, KVCacheType]:
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        Use the model's num_heads and head_dim (5B has 24 heads, 14B has 40).
        """
        num_heads = self.model.num_heads
        head_dim = self.model.dim // num_heads
        crossattn_cache: KVCacheType = []
        crossattn_cache_neg: KVCacheType = []

        for _ in range(self.model.num_layers):
            crossattn_cache.append(
                torch.zeros([2, batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
            )
            crossattn_cache_neg.append(
                torch.zeros([2, batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
            )

        return crossattn_cache, crossattn_cache_neg

    def _slide_kv_window(self, window_size: int = 2):
        """Sliding window KV cache with RoPE position-shift re-encoding + sink.

        Extracted from commit 503f9fa. Env vars:
          KV_SINK_SIZE (default 1) — leading frames preserved with original RoPE.
          kept_total = window_size * num_frame_per_block.
        """
        if self.kv_cache1 is None:
            return
        frame_seqlen = self.model.blocks[0].self_attn.frame_seqlen if hasattr(self.model.blocks[0].self_attn, 'frame_seqlen') else 880
        npfb = self.num_frame_per_block
        sink_size = int(os.environ.get("KV_SINK_SIZE", "1"))
        keep_frames_target = window_size * npfb

        # Diagnostic: SLIDE_TO_CSF0_ONLY=1 sets csf=0 without wiping cache.
        # csf==0 path in main function will re-create caches via _create_kv_caches,
        # so this tests whether the explicit wipe in SLIDE_EMPTY_CACHE is redundant.
        if os.environ.get("SLIDE_TO_CSF0_ONLY", "0") == "1":
            self.current_start_frame = 0
            self._need_reencode = False
            if self.ip_rank == 0:
                print(f"[KV slide TO_CSF0_ONLY] csf=0, cache untouched (will be recreated)")
            return

        # Diagnostic: SLIDE_KEEP_INITIAL=1 preserves only pos 0 (chunk 0's initial
        # prefill content = initial scene anchor), drops everything after. Doesn't
        # re-init via csf==0 path, so pos 0 is NOT overwritten with mid-task scene.
        if os.environ.get("SLIDE_KEEP_INITIAL", "0") == "1":
            for cache_list in [self.kv_cache1, self.kv_cache_neg]:
                if cache_list is None:
                    continue
                for i in range(len(cache_list)):
                    c = cache_list[i]
                    if c.shape[2] >= frame_seqlen:
                        cache_list[i] = c[:, :, :frame_seqlen, :, :].contiguous()
                    else:
                        shape = list(c.shape); shape[2] = 0
                        cache_list[i] = torch.zeros(shape, dtype=c.dtype, device=c.device)
            # csf = npfb + 1 → next main-diff queries at pos npfb+1, npfb+2.
            # Recondition writes pos 1..npfb right before main diff.
            self.current_start_frame = npfb + 1
            self._need_reencode = True
            if self.ip_rank == 0:
                print(f"[KV slide KEEP_INITIAL] kept pos 0, csf={self.current_start_frame}")
            return

        # Diagnostic: SLIDE_EMPTY_CACHE=1 wipes cache to empty (seq_len=0) on slide
        # while keeping csf counter continuous. Tests whether the "stuck in old task
        # phase" issue is due to stale self-attention KV content, vs. RoPE math itself.
        # SLIDE_RESET_LOW=1 additionally resets csf to a LOW value (npfb) so model
        # sees "early task" positions instead of csf=6-10 (training-OOD "end of task").
        if os.environ.get("SLIDE_EMPTY_CACHE", "0") == "1":
            for cache_list in [self.kv_cache1, self.kv_cache_neg]:
                if cache_list is None:
                    continue
                for i in range(len(cache_list)):
                    c = cache_list[i]
                    shape = list(c.shape); shape[2] = 0
                    cache_list[i] = torch.zeros(shape, dtype=c.dtype, device=c.device)
            if os.environ.get("SLIDE_TO_CSF0", "0") == "1":
                self.current_start_frame = 0  # forces csf==0 path (baseline-like)
                self._need_reencode = False   # csf==0 path already re-encodes; skip _need_reencode branch
            elif os.environ.get("SLIDE_RESET_LOW", "0") == "1":
                self.current_start_frame = npfb  # 2 — low position like baseline
            else:
                self.current_start_frame = keep_frames_target + npfb
            if self.ip_rank == 0:
                print(f"[KV slide EMPTY] cache wiped, csf={self.current_start_frame}")
            return

        sample = None
        for candidate in (self.kv_cache1, self.kv_cache_neg):
            if candidate is not None and len(candidate) > 0 and candidate[0].shape[2] > 0:
                sample = candidate[0]; break
        if sample is None:
            return

        frames_in_cache = sample.shape[2] // frame_seqlen
        kept_total = min(keep_frames_target, frames_in_cache)
        sink_size = max(0, min(sink_size, kept_total))
        kept_recent = kept_total - sink_size
        if kept_total == 0:
            return

        shift = frames_in_cache - kept_recent - sink_size

        if shift == 0 or kept_recent == 0:
            self.current_start_frame = kept_total + npfb
            if self.ip_rank == 0:
                print(f"[KV slide] no-shift: sink={sink_size}, recent={kept_recent}, csf={self.current_start_frame}")
            return

        base_model = getattr(self.model, 'base_model', self.model)
        base_model = getattr(base_model, 'model', base_model)
        freqs0 = base_model.freqs[0]
        if freqs0.device != sample.device:
            freqs0 = freqs0.to(sample.device); base_model.freqs[0] = freqs0
        correction = freqs0[shift].conj()
        d0_complex = correction.shape[0]
        skip_math = os.environ.get("SKIP_REROPE_MATH", "0") == "1"

        sink_len = sink_size * frame_seqlen
        recent_len = kept_recent * frame_seqlen

        for cache_list in [self.kv_cache1, self.kv_cache_neg]:
            if cache_list is None:
                continue
            for i in range(len(cache_list)):
                cache = cache_list[i]
                if cache.shape[2] == 0:
                    continue
                if cache.shape[2] < sink_len + recent_len:
                    continue
                sink_cache = cache[:, :, :sink_len, :, :]
                recent_cache = cache[:, :, -recent_len:, :, :]
                if skip_math:
                    cache_list[i] = torch.cat([sink_cache, recent_cache], dim=2).contiguous()
                    continue
                k = recent_cache[0]; v = recent_cache[1]; k_dtype = k.dtype
                k_fp = k.to(torch.float64).reshape(*k.shape[:-1], -1, 2)
                k_complex = torch.view_as_complex(k_fp)
                k_frame = k_complex[..., :d0_complex] * correction
                k_hw = k_complex[..., d0_complex:]
                k_complex_new = torch.cat([k_frame, k_hw], dim=-1).contiguous()
                k_new = torch.view_as_real(k_complex_new).flatten(-2).to(k_dtype)
                recent_new = torch.stack([k_new, v.contiguous()], dim=0)
                cache_list[i] = torch.cat([sink_cache, recent_new], dim=2).contiguous()

        self.current_start_frame = kept_total + npfb
        if self.ip_rank == 0:
            print(f"[KV slide+rerope+sink] shift={shift}, sink={sink_size}, recent={kept_recent}, csf={self.current_start_frame}")

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def preprocess_image(self, image):
        image = (image * (2 / 255) - 1).permute(0, 1, 4, 2, 3)
        return image

    def encode_prompt(self, input_ids, attention_mask):
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(input_ids, attention_mask)
        prompt_emb = prompt_emb.clone().to(dtype=torch.bfloat16)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def _ensure_vae_on_device(self, ref_tensor):
        """Lazily move the VAE to the correct device/dtype on first use."""
        if not getattr(self, '_vae_device_ready', False):
            self.vae.to(device=ref_tensor.device, dtype=torch.bfloat16)
            self.vae.eval()
            self._vae_device_ready = True

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        self._ensure_vae_on_device(input_video)
        with torch.no_grad():
            latents = self.vae.encode(input_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def encode_image(self, image, num_frames, height, width):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            batch_size = image.shape[0]
            clip_context = self.image_encoder.encode_image(image)
            image_input = image.transpose(1, 2)
            image_zeros = torch.zeros(batch_size, 3, num_frames-1, height, width, dtype=torch.bfloat16, device=self._device)
            self._ensure_vae_on_device(image_input)
            with torch.no_grad():
                y = self.vae.encode(torch.concat([image_input, image_zeros], dim=2))
            # Build mask to match VAE output shape (VAE may use different spatial downsampling, e.g. WanVideoVAE38 uses patch_size=2 -> height/16)
            # y shape is B * 16 * (1+(T-1)/4) * H_latent * W_latent
            num_t = y.shape[2]
            h_latent, w_latent = y.shape[3], y.shape[4]
            msk = torch.zeros(batch_size, 4, num_t, h_latent, w_latent, dtype=y.dtype, device=self._device)
            msk[:, :, 0:1, :, :] = 1
            new_image = y[:, :, 0:1]
            # concat: B * (4+16) * (1+(T-1)/4) * H_latent * W_latent
            y = torch.concat([msk, y], dim=1)
        return clip_context, y, new_image
    
    def prepare_extra_input(self, latents=None):
        return {}

    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming") -> nn.Module:
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = get_peft_model(model, lora_config)
        for param in model.parameters():
            param.data = param.to(torch.float32)
        return model

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        data = action_input 
        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id
        # print("embodiment_id", embodiment_id)
        has_real_action = action_input.has_real_action
        action_mask = action_input.action_mask

        state_features = action_input.state

        actions = action_input.action
        # assert the values of action is in between -1 and 1
        if actions.numel() > 0:
            assert actions.min() >= -1.0 and actions.max() <= 1.0, "actions must be in [-1,1] range"
        videos = data["images"]

        videos = rearrange(videos, "b t h w c -> b c t h w")
        print("videos", videos.shape)
        

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)  # [b, t, c, h, w]
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)  # back to [b, c, t, h, w]
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=self.dtype)
        
        # shape of B * max_length * dim
        prompt_embs = self.encode_prompt(data["text"], data["text_attention_mask"])

        # Wan 5B: resize to target resolution so latent tokens/frame matches DiT. Use config target when set
        # (e.g. 160x320 so latent is 10x20 with VAE38 16x → even H,W, no crop in dynamics loss); else 176x320.
        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)

        latents = self.encode_video(videos, self.tiled, (self.tile_size_height, self.tile_size_width), (self.tile_stride_height, self.tile_stride_width))

        # print("latents shape", latents.shape, self.dtype)
        _, _, num_frames, height, width = videos.shape
        image = videos[:, :, :1].transpose(1, 2)

        clip_feas, ys, _ = self.encode_image(image, num_frames, height, width)

        latents = latents.to(self._device)
        clip_feas = clip_feas.to(self._device)
        ys = ys.to(self._device)
        prompt_embs = prompt_embs.to(self._device)
       
        # Loss
        noise = torch.randn_like(latents)

        # specific to autoregressive 
        noise = noise.transpose(1, 2)
        latents = latents.transpose(1, 2)
        
        # ============ VIDEO TIMESTEP SAMPLING ============
        if self.config.decouple_video_action_noise:
            # Decoupled mode: sample video from Beta distribution biased towards HIGH noise
            video_noise_ratio = self.video_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - video_noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
            noise_mode = "DECOUPLED"
        elif self.config.use_high_noise_emphasis:
            # High noise emphasis mode (coupled): BOTH video and action use Beta distribution
            noise_ratio = self.high_noise_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
            noise_mode = "HIGH_NOISE_EMPHASIS"
        else:
            # Original: uniform sampling over full range
            timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (noise.shape[0], noise.shape[1]))
            noise_mode = "STANDARD"
        
        timestep_id_block = timestep_id[:, 1:].reshape(
                    timestep_id.shape[0], -1, self.num_frame_per_block)
        timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]
        
        if actions.numel() > 0:
            noise_action = torch.randn_like(actions)
            assert actions.shape[1] / (noise.shape[1]-1) == (self.model.num_action_per_block // self.num_frame_per_block), f"actions.shape, {actions.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"
            assert (noise.shape[1]-1) / state_features.shape[1] == (self.num_frame_per_block // self.model.num_state_per_block), f"state_features.shape, {state_features.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"
            
            # ============ ACTION TIMESTEP SAMPLING ============
            if self.config.decouple_video_action_noise:
                # Decoupled: sample action timestep independently with full range
                timestep_action_id = torch.randint(
                    0, 
                    self.scheduler.num_train_timesteps, 
                    (actions.shape[0], actions.shape[1])
                )
                action_mode = "INDEPENDENT"
            else:
                # Original coupled: action timestep derived from video timestep
                timestep_action_id = timestep_id_block.repeat(1, 1, actions.shape[1]//(noise.shape[1]-1))
                timestep_action_id = timestep_action_id.reshape(timestep_action_id.shape[0], -1)
                action_mode = "COUPLED"
            
            # Log noise mode once
            if not self._noise_logged:
                video_mean = timestep_id.float().mean().item()
                action_mean = timestep_action_id.float().mean().item()
                if noise_mode == "DECOUPLED":
                    print(f"[NOISE] Mode={noise_mode} | Video: Beta({self.config.video_noise_beta_alpha},1) mean_t={video_mean:.0f} | Action: {action_mode} Uniform mean_t={action_mean:.0f}")
                elif noise_mode == "HIGH_NOISE_EMPHASIS":
                    print(f"[NOISE] Mode={noise_mode} | Video+Action: Beta({self.config.high_noise_beta_alpha},1) mean_t={video_mean:.0f} | Action: {action_mode}")
                else:
                    print(f"[NOISE] Mode={noise_mode} | Video+Action: Uniform mean_t={video_mean:.0f} | Action: {action_mode}")
                self._noise_logged = True
        else:
            noise_action = None
            timestep_action_id = None
            
        timestep_id_block = timestep_id_block.reshape(timestep_id_block.shape[0], -1)
        timestep_id = torch.concat([timestep_id[:, :1], timestep_id_block], dim=1)
        _, num_frames, num_channels, height, width = noise.shape
        # DiT patch_embedding uses stride (1,2,2), so sequence length is num_frames * (H//2) * (W//2)
        tokens_per_frame = (height // 2) * (width // 2)
        seq_len = num_frames * tokens_per_frame

        timestep = self.scheduler.timesteps[timestep_id].to(self._device)
        noisy_latents = self.scheduler.add_noise(latents.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1]))
        training_target = self.scheduler.training_target(latents, noise, timestep).transpose(1, 2)
        
        if actions.numel() > 0:
            timestep_action = self.scheduler.timesteps[timestep_action_id].to(self._device)
            noisy_actions = self.scheduler.add_noise(
                actions.flatten(0, 1),
                noise_action.flatten(0, 1),
                timestep_action.flatten(0, 1),
            ).unflatten(0, (noise_action.shape[0], noise_action.shape[1]))
            training_target_action = self.scheduler.training_target(actions, noise_action, timestep_action)
        else:
            timestep_action = None
            noisy_actions = None
            training_target_action = None

        # Compute loss
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            if actions.numel() > 0:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(1, 2), timestep=timestep, clip_feature=clip_feas, y=ys, context=prompt_embs, seq_len=seq_len,
                    state=state_features, embodiment_id=embodiment_id,
                    action=noisy_actions, timestep_action=timestep_action, 
                    clean_x=latents.transpose(1, 2),
                )
            else:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(1, 2), timestep=timestep, timestep_action=timestep_action, 
                    clip_feature=clip_feas, y=ys, context=prompt_embs, seq_len=seq_len,
                    state=state_features, embodiment_id=embodiment_id,
                    clean_x=latents.transpose(1, 2),
                )

            # Per-sample dynamics loss
            # DiT patch_embedding uses stride (1,2,2), so output spatial size can be smaller than
            # latent when H or W is odd (e.g. latent 11x20 -> model output 10x20). Crop target to match.
            if training_target.shape != video_noise_pred.shape:
                training_target = training_target[
                    ..., : video_noise_pred.shape[3], : video_noise_pred.shape[4]
                ]
            dynamics_loss_per_sample = torch.nn.functional.mse_loss(
                video_noise_pred.float(), training_target.float(), reduction='none'
            ).mean(dim=(1,3,4))  # shape: [B, ...]

            weight_dynamics = dynamics_loss_per_sample * self.scheduler.training_weight(timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1])).to(self._device)
            weighted_dynamics_loss = weight_dynamics.mean()
            
            if actions.numel() > 0:
                action_loss_per_sample = torch.nn.functional.mse_loss(
                    action_noise_pred.float(), training_target_action.float(), reduction='none'
                ) * action_mask  # shape: [B, ...]
                action_loss_per_sample = has_real_action[:, None].float() * action_loss_per_sample  # apply has_real_action
                weight_action = action_loss_per_sample.mean(dim=2) * self.scheduler.training_weight(
                    timestep_action.flatten(0, 1),
                ).unflatten(0, (noise_action.shape[0], noise_action.shape[1])).to(self._device)
                weighted_action_loss = weight_action.mean()
                loss = weighted_dynamics_loss + weighted_action_loss
            else:
                weighted_action_loss = torch.tensor(0.0, device=self._device)
                loss = weighted_dynamics_loss
            # loss = dynamics_loss_per_sample.mean()

        # Record log
        output_dict = {
            "loss": loss,
            "dynamics_loss": weighted_dynamics_loss,
            "action_loss": weighted_action_loss,
        }

        return BatchFeature(data=output_dict)

    def generate_noise(self, shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise
    
    def _get_caches(
        self, kv_caches_input: list[KVCacheType],
    ) -> list[KVCacheType]:
        if self.ip_size > 1:
            assert self.cfg_scale != 1.0, "cfg_scale must be != 1.0 when ip_size > 1"
            assert len(kv_caches_input) == 2
            if self.ip_rank == 0:
                kv_caches = [kv_caches_input[0]]
            else:
                kv_caches = [kv_caches_input[1]]
        else:
            assert len(kv_caches_input) <= 2
            kv_caches = [kv_caches_input[0]]
            if self.cfg_scale != 1.0:
                kv_caches.append(kv_caches_input[1])
        return kv_caches

    def _prepare_text_inputs(self, data: BatchFeature) -> list[tuple[torch.Tensor, torch.Tensor]]:

        if self.ip_size > 1:
            assert self.cfg_scale != 1.0, "cfg_scale must be != 1.0 when ip_size > 1"
            if self.ip_rank == 0:
                text_inputs = [(data["text"], data["text_attention_mask"])]
            else:
                text_inputs = [(data["text_negative"], data["text_attention_mask_negative"])]
        else:
            text_inputs = [(data["text"], data["text_attention_mask"])]
            if self.cfg_scale != 1.0:
                text_inputs.append((data["text_negative"], data["text_attention_mask_negative"]))
        return text_inputs


    def _run_diffusion_steps(
        self,
        noisy_input: torch.Tensor,
        timestep: torch.Tensor,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        y: torch.Tensor,
        clip_feature: torch.Tensor,
        kv_caches: list[KVCacheType],
        crossattn_caches: list[KVCacheType],
        kv_cache_metadata: dict[str, bool | int],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        predictions = []
        for index, prompt_emb in enumerate(context):
            kv_cache = kv_caches[index]
            crossattn_cache = crossattn_caches[index]
            if not kv_cache_metadata["update_kv_cache"] and self.trt_engine is not None:
                obs_noise_pred, action_noise_pred = self.trt_engine(
                    noisy_input,
                    timestep,
                    action=action,
                    timestep_action=timestep_action,
                    state=state,
                    context=prompt_emb,
                    y=y,
                    clip_feature=clip_feature,
                    kv_cache=kv_cache,
                )
            else:
                obs_noise_pred, action_noise_pred, updated_kv_caches = self.model(
                    noisy_input,
                    timestep,
                    action=action,
                    timestep_action=timestep_action,
                    state=state,
                    embodiment_id=embodiment_id,
                    context=prompt_emb,
                    seq_len=seq_len,
                    y=y,
                    clip_feature=clip_feature,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start_frame=kv_cache_metadata["start_frame"],
                )
                if kv_cache_metadata["update_kv_cache"]:
                    for block_index, updated_kv_cache in enumerate(updated_kv_caches):
                        kv_cache[block_index] = updated_kv_cache.clone()
            obs_noise_pred = obs_noise_pred.clone()
            if action_noise_pred is not None:
                action_noise_pred = action_noise_pred.clone()
            else:
                action_noise_pred = torch.tensor(0.0, device=obs_noise_pred.device) # dummy action noise prediction
            predictions.append((obs_noise_pred, action_noise_pred))
        return self._exchange_predictions(predictions)

    def _exchange_predictions(
        self,
        predictions: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self.ip_size == 1:
            return predictions

        assert len(predictions) == 1
        my_predictions = list(predictions[0])

        other_predictions = [torch.empty_like(pred) for pred in my_predictions]

        send_ops = [
            dist.P2POp(op=dist.isend, tensor=pred, group_peer=(self.ip_rank + 1) % self.ip_size, group=self.ip_group)
            for pred in my_predictions
        ]
        recv_ops = [
            dist.P2POp(op=dist.irecv, tensor=other_pred, group_peer=(self.ip_rank + 1) % self.ip_size, group=self.ip_group)
            for other_pred in other_predictions
        ]
        ops = send_ops + recv_ops

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        output_predictions: list[tuple[torch.Tensor, torch.Tensor] | None] = [None for _ in range(self.ip_size)]
        output_predictions[self.ip_rank] = tuple(my_predictions)
        output_predictions[(self.ip_rank + 1) % self.ip_size] = tuple(other_predictions)
        assert all(isinstance(pred, tuple) for pred in output_predictions)
        return cast(list[tuple[torch.Tensor, torch.Tensor]], output_predictions)
    
    def should_run_model(self, index, current_timestep, prev_predictions):

        if not self.dynamic_cache_schedule:
            return self.dit_step_mask[index]

        # Always run first 2 steps to establish history
        if len(prev_predictions) < 2:
            return True

        if self.skip_countdown > 1:
            self.skip_countdown -= 1
            return False
        elif self.skip_countdown == 1:
            self.skip_countdown = 0 
            return True

        v_last = prev_predictions[-1][1].flatten(1).float()
        v_prev = prev_predictions[-2][1].flatten(1).float()
        sim = torch.nn.functional.cosine_similarity(v_last, v_prev, dim=1).mean()

        thresholds = [0.95, 0.93]
        countdowns = [4, 2]

        for threshold, countdown in zip(thresholds, countdowns):
            if sim > threshold:
                self.skip_countdown = countdown
                return False

        return True

    def lazy_joint_video_action(self, backbone_output: BatchFeature, action_input: BatchFeature, latent_video: torch.Tensor | None = None) -> BatchFeature:
        start_time = time.perf_counter()

        # Tracking time taken on GPU for various operations.
        start_text_encoder_event = torch.cuda.Event(enable_timing=True)
        end_text_encoder_event = torch.cuda.Event(enable_timing=True)
        start_image_encoder_event = torch.cuda.Event(enable_timing=True)
        end_image_encoder_event = torch.cuda.Event(enable_timing=True)
        start_vae_event = torch.cuda.Event(enable_timing=True)
        end_vae_event = torch.cuda.Event(enable_timing=True)
        start_kv_event = torch.cuda.Event(enable_timing=True)
        end_kv_event = torch.cuda.Event(enable_timing=True)
        start_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_inference_steps)]
        end_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_inference_steps)]

        self.set_frozen_modules_to_eval_mode()
        data = action_input 
        
        videos = data["images"]

        embodiment_id = action_input.embodiment_id
        state_features = action_input.state

        videos = rearrange(videos, "b t h w c -> b c t h w")

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            videos = videos.to(dtype=self.dtype)
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)  # [b, t, c, h, w]
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)  # back to [b, c, t, h, w]
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=self.dtype)

        state_features = state_features.to(dtype=torch.bfloat16)
        videos = videos.to(dtype=torch.bfloat16)

        # Wan 5B: same as training — resize to target resolution so latent matches DiT
        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)

        if self.language is None:
            print("language is None, reset current_start_frame to 0")
            self.language = data["text"]
            self.current_start_frame = 0
        elif not torch.equal(self.language, data["text"]):
            print("language changed, reset current_start_frame to 0")
            self.current_start_frame = 0
            self.language = data["text"]
        elif os.environ.get("FORCE_RESET_EVERY_CHUNK", "0") == "1":
            # Diagnostic: force csf=0 reset every chunk regardless of sliding state.
            # Effectively makes every chunk run init prefill (csf==0 path) instead of
            # sliding's recondition path. Tests if "init prefill every chunk" is what
            # makes baseline work. Compute cost should be similar (1 prefill + 16 main).
            print("FORCE_RESET_EVERY_CHUNK=1: reset current_start_frame to 0")
            self.current_start_frame = 0
        elif videos.shape[2] == 1 and not (os.environ.get("USE_KVPRESS", "0") == "1"):
            print("videos.shape[2] == 1, reset current_start_frame to 0")
            self.current_start_frame = 0
        elif self.current_start_frame >= self.model.local_attn_size:
            if os.environ.get("USE_KVPRESS", "0") == "1":
                window_size = int(os.environ.get("KV_WINDOW_SIZE", "4"))
                self._slide_kv_window(window_size=window_size)
                self._need_reencode = True
            else:
                print("current_start_frame >= local_attn_size, reset current_start_frame to 0")
                self.current_start_frame = 0

        # Fix for first-frame distortion in sliding mode: baseline re-encodes CLIP/ys
        # every chunk (via videos.shape[2]==1 reset path), keeping visual conditioning
        # fresh. Sliding mode skips that reset, so without this flag the clip_feas/ys
        # would remain stale from chunk 0 and the model's first predicted frame would
        # carry the initial-frame color bias. Setting _need_reencode=True here forces
        # the encode_image refresh in the block below for every non-reset sliding chunk.
        if os.environ.get("USE_KVPRESS", "0") == "1" and self.current_start_frame != 0:
            self._need_reencode = True

        if self.ip_rank == 0:
            print("videos shape", videos.shape, self.num_frames)

        start_text_encoder_event.record()

        text_inputs = self._prepare_text_inputs(data)
        prompt_embs = [self.encode_prompt(text, attention_mask) for text, attention_mask in text_inputs]

        end_text_encoder_event.record()
        
        start_image_encoder_event.record()

        _, _, num_frames, height, width = videos.shape
        if videos.shape[2] == 4 or videos.shape[2] == 9:
            # special case for real-world eval where language is updated
            image = videos[:, :, -1:].transpose(1, 2)
        else:
            image = videos[:, :, :1].transpose(1, 2)

        if self.current_start_frame == 0:
            clip_feas, ys, image = self.encode_image(image, self.num_frames, height, width)
            self.clip_feas = clip_feas.to(dtype=image.dtype)
            self.ys = ys.to(dtype=image.dtype)
        elif getattr(self, '_need_reencode', False):
            # After KV sliding window: refresh CLIP/ys from current obs
            # (KV cache kept, but visual conditioning must update). Also save the
            # single-frame latest-obs latent (encode_image's 3rd return) so the
            # output prepend at end of forward() can use the CURRENT latest frame
            # as the saved-video anchor instead of image[:, :1] (which would be
            # the oldest of the multi-frame VAE encode latents — frame N-8 in the
            # 9-frame buffer, ~64 sim-steps stale).
            clip_feas, ys, latest_obs_latent = self.encode_image(image, self.num_frames, height, width)
            self.clip_feas = clip_feas.to(dtype=image.dtype)
            self.ys = ys.to(dtype=image.dtype)
            self._latest_obs_latent = latest_obs_latent.transpose(1, 2)  # (B, 1, C, H', W')
            self._need_reencode = False
            if self.ip_rank == 0:
                print("[KV slide] re-encoded CLIP/ys from current observation")

        assert self.clip_feas is not None and self.ys is not None, "clip_feas and ys must be set"

        end_image_encoder_event.record()

        start_vae_event.record()

        if latent_video is not None and self.current_start_frame != 0:
            image = latent_video
            if self.ip_rank == 0:
                print("image shape@@", image.shape)
        elif self.current_start_frame != 0:
            # this is for real world execution
            target_raw_frames = self.num_frame_per_block * 4 + 1  # e.g. 9
            num_input_frames = videos.shape[2]
            if (num_input_frames - 1) // 4 == self.num_frame_per_block:
                print("no further action")
            elif num_input_frames < target_raw_frames:
                # Few frames (e.g. single obs when USE_KVPRESS=1 carries KV across
                # chunks). Repeat the last frame to fill the required count.
                videos = videos[:, :, -1:].repeat(1, 1, target_raw_frames, 1, 1)
            elif videos.shape[2] // 4 != self.num_frame_per_block:
                # Repeating videos along dim 2.
                repeat_factor = self.num_frame_per_block // (videos.shape[2] // 4)
                videos = torch.repeat_interleave(videos, repeat_factor, dim=2)
                if os.environ.get("DUP_FIRST_FRAME", "1") == "1":
                    first_frame = videos[:, :, 0:1]
                    videos = torch.cat([first_frame, videos], dim=2)
            else:
                if os.environ.get("DUP_FIRST_FRAME", "1") == "1":
                    first_frame = videos[:, :, 0:1]
                    videos = torch.cat([first_frame, videos], dim=2)
           
            image = self.vae.encode(
                videos,
                tiled=self.tiled,
                tile_size=(self.tile_size_height, self.tile_size_width),
                tile_stride=(self.tile_stride_height, self.tile_stride_width),
            )

        end_vae_event.record()

        noise_obs = self.generate_noise((image.shape[0], image.shape[1], self.num_frame_per_block, image.shape[3], image.shape[4]), seed=self.seed, device='cuda', dtype=torch.bfloat16)
        noise_action = self.generate_noise((image.shape[0], self.action_horizon, self.model.action_dim), seed=self.seed, device='cuda', dtype=torch.bfloat16)
        batch_size, num_channels, num_frames, height, width = noise_obs.shape
        ######### Generate video #########
        # DiT patch_embedding uses stride (1,2,2), so tokens per frame = (H//2)*(W//2)
        tokens_per_frame = (height // 2) * (width // 2)
        frame_seqlen = tokens_per_frame
        seq_len = num_frames * frame_seqlen

        image = image.transpose(1, 2)
        noise_obs = noise_obs.transpose(1, 2)

        if self.current_start_frame == 0:
            # Reinitialize KV cache and crossattn cache for each new sequence.
            self.kv_cache1, self.kv_cache_neg = self._create_kv_caches(
                batch_size=batch_size,
                dtype=noise_obs.dtype,
                device=noise_obs.device,
                frame_seqlen=frame_seqlen,
            )
            self.crossattn_cache, self.crossattn_cache_neg = self._create_crossattn_caches(
                batch_size=batch_size,
                dtype=noise_obs.dtype,
                device=noise_obs.device,
            )

        assert self.kv_cache1 is not None
        assert self.kv_cache_neg is not None
        assert self.crossattn_cache is not None
        assert self.crossattn_cache_neg is not None
        kv_caches = self._get_caches(
            [self.kv_cache1, self.kv_cache_neg],
        )
        crossattn_caches = self._get_caches(
            [self.crossattn_cache, self.crossattn_cache_neg],
        )

        start_kv_event.record()

        # Track whether multi-frame prefill ran this call (so we can skip the
        # redundant recondition that would overwrite the same cache positions).
        _did_multiframe_prefill = False
        if self.current_start_frame == 0:
            # Multi-frame init prefill (MULTIFRAME_PREFILL=1): when caller has
            # buffered the past N raw frames (videos.shape[2] >= 4), VAE-encode
            # the full window (9 raw → 3 latent) and prefill cache positions
            # 0..N-1 with REAL recent observation history instead of only the
            # current single frame. Default (no env var) keeps the original
            # 1-frame init prefill behaviour.
            n_frames_prefill = 1
            # noisy_input format expected by model: (B, C, T, H', W')
            # `image` post line 1282 transpose is (B, 1, C, H', W'), so transpose
            # back to (B, C, 1, H', W') for the single-frame default path.
            prefill_obs = image.transpose(1, 2)
            if (os.environ.get("MULTIFRAME_PREFILL", "0") == "1"
                    and videos.shape[2] >= 4):
                target_raw_frames = self.num_frame_per_block * 4 + 1  # 9
                num_input_frames = videos.shape[2]
                if (num_input_frames - 1) // 4 == self.num_frame_per_block:
                    adj_videos = videos
                elif num_input_frames < self.num_frame_per_block * 2:
                    # Truly degenerate: <4 raw frames buffered. Fall back to
                    # last-frame replication (acts as single-frame prefill).
                    adj_videos = videos[:, :, -1:].repeat(1, 1, target_raw_frames, 1, 1)
                elif videos.shape[2] // 4 != self.num_frame_per_block:
                    repeat_factor = self.num_frame_per_block // (videos.shape[2] // 4)
                    adj_videos = torch.repeat_interleave(videos, repeat_factor, dim=2)
                    first_frame = adj_videos[:, :, 0:1]
                    adj_videos = torch.cat([first_frame, adj_videos], dim=2)
                else:
                    first_frame = videos[:, :, 0:1]
                    adj_videos = torch.cat([first_frame, videos], dim=2)
                multi_image = self.vae.encode(
                    adj_videos,
                    tiled=self.tiled,
                    tile_size=(self.tile_size_height, self.tile_size_width),
                    tile_stride=(self.tile_stride_height, self.tile_stride_width),
                )
                # multi_image: (B, C, T_latent, H', W'), T_latent = 1+(9-1)/4 = 3.
                prefill_obs = multi_image
                n_frames_prefill = multi_image.shape[2]
                # NOTE: do NOT overwrite `image` here. `image` holds the
                # single-frame latent of the LATEST observation (from
                # encode_image at line 1219, anchored on videos[:, :, -1:] for
                # shape==9). It is reused at the end of forward() to prepend a
                # "current scene" anchor frame to the output video. Overwriting
                # it with multi_image[:, 0] (oldest frame) would make the saved
                # pred video start with a stale frame from N sim-steps ago.
                # Recondition is skipped via _did_multiframe_prefill, so we
                # don't need image to hold the multi-frame latents anyway.
                _did_multiframe_prefill = True
                if self.ip_rank == 0:
                    print(f"[MULTIFRAME_PREFILL] {n_frames_prefill} latent frames from {num_input_frames}-frame buffer")
            timestep = torch.ones([batch_size, n_frames_prefill], device=noise_obs.device, dtype=torch.int64) * 0
            self._run_diffusion_steps(
                noisy_input=prefill_obs,
                timestep=timestep * 0,
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                context=prompt_embs,
                seq_len=frame_seqlen * n_frames_prefill,
                y=self.ys[:, :, 0:n_frames_prefill],
                clip_feature=self.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(
                    start_frame=0,
                    update_kv_cache=True,
                ),
            )
            self.current_start_frame += n_frames_prefill

        timestep = torch.ones([batch_size, self.num_frame_per_block], device=noise_obs.device, dtype=torch.int64) * 0

        # SLIDE_SKIP_RECONDITION=1: in sliding mode, skip the recondition block
        # entirely. Baseline never runs it (always csf==1 here), so this makes
        # sliding match baseline's "no recondition overwrite" behavior.
        _skip_recond = (os.environ.get("USE_KVPRESS", "0") == "1"
                        and os.environ.get("SLIDE_SKIP_RECONDITION", "0") == "1")
        # When multi-frame prefill just populated positions 0..N-1 with real
        # observation latents, a recondition pass would redundantly overwrite
        # the last npfb of those positions. Skip it.
        if self.current_start_frame != 1 and not _skip_recond and not _did_multiframe_prefill:
            current_ref_latents = image[:, -self.num_frame_per_block:]
            # SLIDE_RECOND_Y_HEAD=1: use ys[:, :, 0:npfb] (pos 0 has msk=1 real
            # signal + pos 1 zero-pad) instead of the zero-pad-only slice at
            # csf-npfb:csf. Brings recondition y closer to training distribution.
            if (os.environ.get("USE_KVPRESS", "0") == "1"
                    and os.environ.get("SLIDE_RECOND_Y_HEAD", "0") == "1"):
                y = self.ys[:, :, 0:self.num_frame_per_block]
            elif self.current_start_frame <= self.ys.shape[2]:
                y = self.ys[:, :, self.current_start_frame - self.num_frame_per_block : self.current_start_frame]
            else:
                y = self.ys[:, :, -self.num_frame_per_block:]
            self._run_diffusion_steps(
                noisy_input=current_ref_latents.transpose(1, 2),
                timestep=timestep * 0,
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                context=prompt_embs,
                seq_len=seq_len,
                y=y,
                clip_feature=self.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(
                    start_frame=self.current_start_frame - self.num_frame_per_block,
                    update_kv_cache=True,
                ),
            )

        end_kv_event.record()

        noisy_input = noise_obs
        noisy_input_action = noise_action

        # Step 3.1: Spatial denoising loop

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler_action = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(
            self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift)
        sample_scheduler_action.set_timesteps(
            self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift)

        # Decoupled inference: video sigmas end at video_final_noise instead of 0
        # This rescales the schedule so video still takes all denoising steps, 
        # but ends at a higher noise level (e.g., 1.0 → 0.9 → 0.8 instead of 1.0 → 0.5 → 0.0)
        if self.config.decouple_inference_noise:
            video_final_noise = self.config.video_inference_final_noise
            # Rescale video sigmas: map [sigma_max, 0] -> [sigma_max, video_final_noise]
            sigma_max = sample_scheduler.sigmas[0].item()
            sample_scheduler.sigmas = sample_scheduler.sigmas * (sigma_max - video_final_noise) / sigma_max + video_final_noise
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64)
            if self.ip_rank == 0:
                print(f"Decoupled inference: video sigmas {sigma_max:.3f} -> {sample_scheduler.sigmas[-1].item():.3f}")

        start_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in sample_scheduler.timesteps]
        end_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in sample_scheduler.timesteps]
        prev_predictions = [] 
        self.skip_countdown = 0
        dit_compute_steps = 0
        for index, current_timestep in enumerate(sample_scheduler.timesteps):
            start_diffusion_events[index].record()

            # Get timesteps from respective schedulers
            action_timestep = sample_scheduler_action.timesteps[index]
            video_timestep = sample_scheduler.timesteps[index]  # Already rescaled if decoupled

            # set current timestep
            timestep = torch.ones(
                [batch_size, self.num_frame_per_block],
                device=noise_obs.device,
                dtype=torch.int64,
            ) * video_timestep
            timestep_action = torch.ones(
                [batch_size, self.action_horizon],
                device=noise_obs.device,
                dtype=torch.int64,
            ) * action_timestep

            # check if we need to run the DIT step
            should_run_model = self.should_run_model(index, current_timestep, prev_predictions)
            if should_run_model:
                dit_compute_steps += 1
                if self.current_start_frame + self.num_frame_per_block <= self.ys.shape[2]:
                    y = self.ys[:, :, self.current_start_frame : self.current_start_frame + self.num_frame_per_block]
                else:
                    y = self.ys[:, :, -self.num_frame_per_block:]
                predictions = self._run_diffusion_steps(
                    noisy_input=noisy_input.transpose(1, 2),
                    timestep=timestep,
                    action=noisy_input_action,
                    timestep_action=timestep_action,
                    state=state_features,
                    embodiment_id=embodiment_id,
                    context=prompt_embs,
                    seq_len=seq_len,
                    y=y,
                    clip_feature=self.clip_feas,
                    kv_caches=kv_caches,
                    crossattn_caches=crossattn_caches,
                    kv_cache_metadata=dict(
                        start_frame=self.current_start_frame,
                        update_kv_cache=False,
                    ),
                )
                flow_pred_cond, flow_pred_cond_action = predictions[0]
                flow_pred_uncond, flow_pred_uncond_action = predictions[1]

                flow_pred = flow_pred_uncond + self.cfg_scale * (flow_pred_cond - flow_pred_uncond)
                prev_predictions.append((current_timestep, flow_pred, flow_pred_cond_action))
                max_cache_size = 2
                if len(prev_predictions) > max_cache_size:
                    prev_predictions.pop(0)

            else:
                assert len(prev_predictions) > 0, "prev_predictions must be set when skipping"
                _, flow_pred, flow_pred_cond_action = prev_predictions[-1]

            end_diffusion_events[index].record()

            # Video: denoising step (uses rescaled schedule if decoupled)
            noisy_input = sample_scheduler.step(
                model_output=flow_pred.transpose(1, 2),
                timestep=video_timestep,
                sample=noisy_input,
                step_index=index,
                return_dict=False,
            )[0]
            
            # Action: always fully denoises with standard schedule (1000->0)
            noisy_input_action = sample_scheduler_action.step(
                model_output=flow_pred_cond_action,
                timestep=action_timestep,
                sample=noisy_input_action,
                step_index=index,
                return_dict=False,
            )[0]

        latents = noisy_input
        latents_action = noisy_input_action
        output = latents

        if self.current_start_frame == 1:
            output = torch.cat([image, output], dim=1)
        elif os.environ.get("USE_KVPRESS", "0") == "1" and self.current_start_frame != 0:
            # Sliding mode: prepend the LATEST single-frame observation latent as
            # video anchor. In csf!=0 path, `image` was overwritten by vae.encode
            # of the full 9-frame buffer → 3 latents, where image[:, :1] is the
            # OLDEST frame's encoding (~64 sim-steps stale). The decoded saved
            # pred video would start with that old frame and the temporal jump
            # to the new prediction creates incoherent first frames. Use the
            # latest_obs_latent saved from encode_image instead (the actual
            # current observation).
            anchor = getattr(self, '_latest_obs_latent', None)
            if anchor is None:
                anchor = image[:, :1]
            output = torch.cat([anchor, output], dim=1)
        self.current_start_frame += self.num_frame_per_block

        # Do torch.cuda.synchronize() to ensure all operations are completed before timing.
        # This isn't expected to affect inference performance since it's at the end of an inference step.
        torch.cuda.synchronize()

        total_time = time.perf_counter() - start_time
        text_encoder_time = start_text_encoder_event.elapsed_time(end_text_encoder_event) / 1000
        image_encoder_time = start_image_encoder_event.elapsed_time(end_image_encoder_event) / 1000
        vae_time = start_vae_event.elapsed_time(end_vae_event) / 1000
        kv_creation_time = start_kv_event.elapsed_time(end_kv_event) / 1000
        diffusion_times = [s.elapsed_time(e) for s, e in zip(start_diffusion_events, end_diffusion_events)]
        diffusion_time = sum(diffusion_times) / 1000
        scheduler_time = total_time - kv_creation_time - diffusion_time - text_encoder_time - image_encoder_time - vae_time

        if self.ip_rank == 0:
            print(f"Time taken: Total {total_time:.2f} seconds, "
                  f"Text Encoder {text_encoder_time:.2f} seconds, "
                  f"Image Encoder {image_encoder_time:.2f} seconds, "
                  f"VAE {vae_time:.2f} seconds, "
                  f"KV Cache Creation {kv_creation_time:.2f} seconds, "
                  f"Diffusion {diffusion_time:.2f} seconds, "
                  f"DIT Compute Steps {dit_compute_steps} steps, "
                  f"Scheduler {scheduler_time:.2f} seconds")

        return BatchFeature(data={"action_pred": latents_action, "video_pred": output.transpose(1, 2)})
    
    def cache_predict_order1(self, current_timestep, timestep_1, f1, timestep_2, f2):
        h_curr = current_timestep - timestep_1
        h_past = timestep_1 - timestep_2

        v_prime = (f1 - f2) / h_past

        # Prediction 
        damping_factor = 0.25
        flow_pred = f1 + (v_prime * h_curr) * damping_factor
        return flow_pred

    def post_initialize(self):
        # Move models to the cuda device and set the dtype to bfloat16.
        print("Moving models to the cuda device and setting the dtype to bfloat16.")
        self.model.to(device=self._device, dtype=torch.bfloat16)
        self.text_encoder.to(device=self._device, dtype=torch.bfloat16)
        self.image_encoder.to(device=self._device, dtype=torch.bfloat16)
        self.vae.to(device=self._device, dtype=torch.bfloat16)
        import os
        ENABLE_TENSORRT = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"
        LOAD_TRT_ENGINE = os.getenv("LOAD_TRT_ENGINE", None)

        # Torch compile the modules. Skip _forward_blocks: Dynamo with fullgraph can fail on
        # shape variation (e.g. x [1,50,C] vs e [1,200,C]); the block aligns e to x at runtime.
        if not ENABLE_TENSORRT:
            print("Torch compiling the TextEncoder, ImageEncoder, and VAE modules (Wan _forward_blocks not compiled).")

            self.text_encoder.forward = torch.compile(
                mode="reduce-overhead", fullgraph=True, dynamic=False,
            )(self.text_encoder.forward)

            self.image_encoder.model.visual.forward = torch.compile(
                mode="reduce-overhead", fullgraph=True, dynamic=False,
            )(self.image_encoder.model.visual.forward)

            self.vae.model.encode = torch.compile(
                mode="reduce-overhead", fullgraph=True, dynamic=False,
            )(self.vae.model.encode)
        
        self.trt_engine = None
        if LOAD_TRT_ENGINE is not None:
            print(f"Loading TRT engine from {LOAD_TRT_ENGINE}")
            import groot.control.tensorrt_utils as trt_utils
            model_path = LOAD_TRT_ENGINE
            self.trt_engine = trt_utils.load_tensorrt_engine(model_path, model_type="ar_14B")

    def parallelize(self, device_mesh: DeviceMesh) -> None:
        ip_mesh = device_mesh["ip"]
        self.ip_rank = ip_mesh.get_local_rank()
        self.ip_size = ip_mesh.size()
        self.ip_group = ip_mesh.get_group()

        assert self.ip_size == 1 or self.ip_size == 2, "ip_size must be 1 or 2"
        assert self.ip_rank >= 0 and self.ip_rank < self.ip_size, "ip_rank must be in [0, ip_size)"

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
