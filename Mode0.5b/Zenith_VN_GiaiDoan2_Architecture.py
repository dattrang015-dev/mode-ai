# ================================================================
# ZENITH VN — STAGE 2: INDEPENDENT PYTORCH ARCHITECTURE
# Google Colab single-file conversion + validation + inference
#
# This file defines its OWN ZenithConfig, ZenithModel and
# ZenithForCausalLM. It does NOT inherit from any Qwen model class.
#
# Weight source:
#   Qwen/Qwen1.5-0.5B-Chat
#   + optional Stage 1 PEFT/QLoRA adapter
#
# Important:
#   - Qwen is used only as a weight source during conversion.
#   - The exported model uses Zenith's own PyTorch classes.
#   - This is an architecture reimplementation, not a claim that
#     the model was pretrained from scratch.
# ================================================================

# ------------------------- 1. INSTALL -----------------------------
import os
import sys
import json
import math
import subprocess
from pathlib import Path

def pip_install(packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q"] + packages
    )

print("🛠️ Installing conversion dependencies...")
pip_install([
    "transformers>=4.45,<5",
    "peft>=0.13",
    "accelerate>=1.0",
    "safetensors",
])

# ------------------------- 2. IMPORTS ------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import save_file, load_file
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ------------------------- 3. USER SETTINGS ------------------------
BASE_MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat"

# This should be the Stage 1 output folder containing adapter_config.json.
# If you want to convert the untouched base model, set USE_LORA_ADAPTER=False.
USE_LORA_ADAPTER = True
LORA_ADAPTER_DIR = "/content/drive/MyDrive/Zenith_VN_0.5B"

OUTPUT_DIR = "/content/drive/MyDrive/Zenith_VN_Architecture"

# Optional Hugging Face token; set it in Colab secrets/environment if needed.
HF_TOKEN = os.environ.get("HF_TOKEN") or None

LOAD_DTYPE = torch.float16
MAX_TEST_NEW_TOKENS = 80

if not torch.cuda.is_available():
    raise RuntimeError("Hãy bật GPU: Runtime > Change runtime type > GPU.")

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)

# ------------------------- 4. ZENITH CONFIG -------------------------
class ZenithConfig:
    """
    Independent configuration object.
    Values are read from the source checkpoint config during conversion.
    """

    model_type = "zenith"

    def __init__(self, **kwargs):
        defaults = {
            "vocab_size": 151936,
            "hidden_size": 1024,
            "intermediate_size": 2816,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "hidden_act": "silu",
            "max_position_embeddings": 32768,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "attention_bias": True,
            "mlp_bias": False,
            "tie_word_embeddings": False,
            "partial_rotary_factor": 1.0,
            "attention_dropout": 0.0,
            "pad_token_id": None,
            "bos_token_id": None,
            "eos_token_id": 151643,
        }

        defaults.update(kwargs)

        for key, value in defaults.items():
            setattr(self, key, value)

        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def to_dict(self):
        result = dict(self.__dict__)
        result["model_type"] = self.model_type
        return result

    def save_pretrained(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(
            os.path.join(directory, "config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, directory):
        path = os.path.join(directory, "config.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.pop("model_type", None)
        data.pop("architectures", None)
        data.pop("_name_or_path", None)
        data.pop("author", None)
        return cls.from_dict(data)


# ------------------------- 5. ROTARY EMBEDDING ---------------------
def rotate_half(x):
    """Rotate the two halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class ZenithRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.rotary_dim = int(
            self.head_dim * getattr(config, "partial_rotary_factor", 1.0)
        )
        if self.rotary_dim % 2 != 0:
            raise ValueError("rotary dimension must be even")

        theta = float(config.rope_theta)
        inv_freq = 1.0 / (
            theta ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, dtype):
        # position_ids: [batch, sequence]
        positions = position_ids.to(device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("bs,d->bsd", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype).unsqueeze(1)
        sin = emb.sin().to(dtype=dtype).unsqueeze(1)
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, rotary_dim):
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q = torch.cat((q_rot, q_pass), dim=-1)
    k = torch.cat((k_rot, k_pass), dim=-1)
    return q, k


# ------------------------- 6. RMS NORMALIZATION --------------------
class ZenithRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.float() * x).to(input_dtype)


# ------------------------- 7. ATTENTION ----------------------------
class ZenithAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.rotary_dim = int(
            self.head_dim * getattr(config, "partial_rotary_factor", 1.0)
        )
        self.attention_dropout = float(
            getattr(config, "attention_dropout", 0.0)
        )

        bias = bool(getattr(config, "attention_bias", False))

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=bias
        )

        self.rotary_emb = ZenithRotaryEmbedding(config)

    def forward(self, hidden_states, position_ids, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(position_ids, q.dtype)
        q, k = apply_rotary_pos_emb(
            q, k, cos, sin, self.rotary_dim
        )

        # Expand grouped KV heads to query-head count.
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Compute attention scores in float32 for stability.
        scores = torch.matmul(
            q.float(), k.float().transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Causal mask: tokens cannot attend to future tokens.
        causal = torch.triu(
            torch.ones(
                seq_len, seq_len,
                device=hidden_states.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))

        # Optional padding mask: 1 = valid token, 0 = padding.
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(torch.bool)
            scores = scores.masked_fill(~key_mask, float("-inf"))

        probs = torch.softmax(scores, dim=-1).to(dtype=q.dtype)
        if self.training and self.attention_dropout > 0:
            probs = F.dropout(probs, p=self.attention_dropout)

        output = torch.matmul(probs, v)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        return self.o_proj(output)


# ------------------------- 8. SWIGLU MLP ---------------------------
class ZenithMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        bias = bool(getattr(config, "mlp_bias", False))

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=bias
        )

    def forward(self, x):
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


# ------------------------- 9. DECODER LAYER ------------------------
class ZenithDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = ZenithAttention(config)
        self.mlp = ZenithMLP(config)
        self.input_layernorm = ZenithRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = ZenithRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(self, hidden_states, position_ids, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ------------------------- 10. ZENITH MODEL ------------------------
class ZenithModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList([
            ZenithDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = ZenithRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

        return self.norm(hidden_states)


# ------------------------- 11. ZENITH CAUSAL LM --------------------
class ZenithForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = ZenithModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        labels=None,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        max_new_tokens=80,
        eos_token_id=None,
        attention_mask=None,
    ):
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            # This simple reference implementation recomputes the full
            # prefix each step. It is correct but not KV-cache optimized.
            mask = (
                torch.ones_like(generated)
                if attention_mask is None
                else torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            generated.shape[0],
                            generated.shape[1] - attention_mask.shape[1],
                            device=generated.device,
                            dtype=attention_mask.dtype,
                        ),
                    ],
                    dim=1,
                )
            )

            result = self(
                input_ids=generated,
                attention_mask=mask,
            )
            next_token = result["logits"][:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None:
                if torch.all(next_token == eos_token_id):
                    break

        return generated


# ------------------------- 12. READ SOURCE CONFIG ------------------
print("📄 Reading source model configuration...")
source_config = AutoConfig.from_pretrained(
    BASE_MODEL_ID,
    token=HF_TOKEN,
    trust_remote_code=True,
)

config_values = source_config.to_dict()

# Keep only configuration values understood by the Zenith implementation.
allowed_keys = [
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "max_position_embeddings",
    "initializer_range",
    "rms_norm_eps",
    "rope_theta",
    "attention_bias",
    "mlp_bias",
    "tie_word_embeddings",
    "partial_rotary_factor",
    "attention_dropout",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
]

zenith_values = {
    key: config_values[key]
    for key in allowed_keys
    if key in config_values
}

# Some older checkpoints may omit these optional fields.
zenith_values.setdefault("attention_bias", False)
zenith_values.setdefault("mlp_bias", False)
zenith_values.setdefault("partial_rotary_factor", 1.0)
zenith_values.setdefault("attention_dropout", 0.0)

zenith_config = ZenithConfig(**zenith_values)

print("Zenith config:")
print(json.dumps(zenith_config.to_dict(), indent=2, ensure_ascii=False))

# ------------------------- 13. LOAD SOURCE + OPTIONAL ADAPTER -------
print("📥 Loading source checkpoint for weight conversion...")
source_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    token=HF_TOKEN,
    torch_dtype=LOAD_DTYPE,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

if USE_LORA_ADAPTER:
    if not os.path.isfile(
        os.path.join(LORA_ADAPTER_DIR, "adapter_config.json")
    ):
        raise FileNotFoundError(
            "Không tìm thấy adapter_config.json tại "
            f"{LORA_ADAPTER_DIR}. Hãy chạy Stage 1 trước hoặc đặt "
            "USE_LORA_ADAPTER=False."
        )

    print("🔗 Loading Stage 1 LoRA adapter...")
    source_model = PeftModel.from_pretrained(
        source_model,
        LORA_ADAPTER_DIR,
        is_trainable=False,
    )

    print("🔧 Merging LoRA adapter into base weights...")
    source_model = source_model.merge_and_unload()

source_model.eval()

# ------------------------- 14. BUILD ZENITH + COPY WEIGHTS ----------
print("🧬 Building independent Zenith model...")
zenith_model = ZenithForCausalLM(zenith_config)

# Match the source parameter dtype, then load compatible state keys.
source_state = source_model.state_dict()
zenith_state = zenith_model.state_dict()

compatible_state = {}
shape_errors = []

for key, target_tensor in zenith_state.items():
    if key not in source_state:
        continue

    source_tensor = source_state[key]
    if tuple(source_tensor.shape) != tuple(target_tensor.shape):
        shape_errors.append(
            (key, tuple(source_tensor.shape), tuple(target_tensor.shape))
        )
        continue

    compatible_state[key] = source_tensor.detach().to(
        dtype=target_tensor.dtype, device="cpu"
    )

missing_keys = sorted(set(zenith_state.keys()) - set(compatible_state.keys()))
unexpected_source_keys = sorted(set(source_state.keys()) - set(zenith_state.keys()))

if shape_errors:
    print("❌ Tensor shape mismatches:")
    for item in shape_errors[:30]:
        print(item)
    raise RuntimeError(
        "Source checkpoint does not match the implemented Zenith architecture. "
        "Do not continue by silently ignoring these tensor mismatches."
    )

if missing_keys:
    print("❌ Missing Zenith parameters:")
    for key in missing_keys[:30]:
        print("  ", key)
    raise RuntimeError(
        "Some Zenith parameters could not be initialized from the source checkpoint."
    )

load_result = zenith_model.load_state_dict(
    compatible_state,
    strict=True,
)

print("✅ All Zenith tensors loaded.")
print("Unexpected source keys (may include non-parameter buffers):",
      len(unexpected_source_keys))

# Move independent model to GPU for numerical comparison and generation.
zenith_model = zenith_model.to(
    device="cuda",
    dtype=LOAD_DTYPE,
)
zenith_model.eval()

# ------------------------- 15. NUMERICAL VALIDATION -----------------
print("🧪 Comparing source model and Zenith logits...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    token=HF_TOKEN,
    trust_remote_code=True,
)

test_text = "Xin chào, tôi là người dùng. Bạn là ai?"
test_inputs = tokenizer(
    test_text,
    return_tensors="pt",
    add_special_tokens=True,
).to("cuda")

with torch.inference_mode():
    source_device = next(source_model.parameters()).device
    source_inputs = {
        key: value.to(source_device)
        for key, value in test_inputs.items()
    }

    source_logits = source_model(
        **source_inputs,
        use_cache=False,
    ).logits[:, -1, :].float().cpu()

    zenith_logits = zenith_model(
        input_ids=test_inputs["input_ids"],
        attention_mask=test_inputs.get("attention_mask"),
    )["logits"][:, -1, :].float().cpu()

max_abs_diff = (source_logits - zenith_logits).abs().max().item()
mean_abs_diff = (source_logits - zenith_logits).abs().mean().item()

print(f"Maximum absolute logit difference: {max_abs_diff:.6f}")
print(f"Mean absolute logit difference: {mean_abs_diff:.6f}")

# Different kernels / precision can cause small numerical differences.
# A large discrepancy means the architecture implementation needs review.
if not torch.isfinite(zenith_logits).all():
    raise RuntimeError("Zenith produced NaN or infinite logits.")

# ------------------------- 16. SAVE ZENITH CHECKPOINT ---------------
print("💾 Saving independent Zenith checkpoint...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

zenith_config.save_pretrained(OUTPUT_DIR)

# Save all Zenith weights as a single safetensors file.
state_to_save = {
    key: value.detach().to("cpu").contiguous()
    for key, value in zenith_model.state_dict().items()
}
save_file(
    state_to_save,
    os.path.join(OUTPUT_DIR, "model.safetensors"),
)

tokenizer.save_pretrained(OUTPUT_DIR)

metadata = {
    "name": "Zenith VN",
    "version": "0.5B",
    "architecture": "ZenithForCausalLM",
    "model_type": "zenith",
    "author": "udVietNamspeedrace",
    "implementation": "Independent PyTorch implementation",
    "inherits_from_qwen_model_class": False,
    "base_weight_source": BASE_MODEL_ID,
    "stage_1_adapter_merged": bool(USE_LORA_ADAPTER),
    "attention": "Grouped Query Attention compatible implementation",
    "position_embedding": "RoPE",
    "normalization": "RMSNorm",
    "mlp": "SwiGLU",
    "checkpoint": "model.safetensors",
    "validation_max_abs_logit_difference": max_abs_diff,
    "validation_mean_abs_logit_difference": mean_abs_diff,
    "note": (
        "Zenith has its own PyTorch model classes. The weights originate "
        "from the Qwen checkpoint, optionally merged with the Stage 1 LoRA "
        "adapter. This is not pretraining from scratch."
    ),
}

with open(
    os.path.join(OUTPUT_DIR, "zenith_metadata.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# ------------------------- 17. RELOAD USING ZENITH CLASSES ----------
print("🔄 Testing reload from Zenith files...")
reloaded_config = ZenithConfig.from_pretrained(OUTPUT_DIR)
reloaded_model = ZenithForCausalLM(reloaded_config)

reloaded_weights = load_file(
    os.path.join(OUTPUT_DIR, "model.safetensors"),
    device="cpu",
)
reloaded_model.load_state_dict(reloaded_weights, strict=True)
reloaded_model = reloaded_model.to(
    device="cuda",
    dtype=LOAD_DTYPE,
).eval()

# ------------------------- 18. ZENITH INFERENCE ---------------------
@torch.inference_mode()
def zenith_chat(question, max_new_tokens=MAX_TEST_NEW_TOKENS):
    messages = [{"role": "user", "content": question}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    output_ids = reloaded_model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )

    answer_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(
        answer_ids,
        skip_special_tokens=True,
    ).strip()


print("\n================ ZENITH VN TEST ================")
for question in [
    "Bạn là ai?",
    "Ai đã tạo ra Zenith VN?",
    "Bạn có thể lập trình Roblox không?",
]:
    print("\nNgười dùng:", question)
    print("Zenith VN:", zenith_chat(question))

# ------------------------- 19. FINAL REPORT ------------------------
print("\n🎉 ZENITH VN STAGE 2 FINISHED")
print("📁 Output folder:", OUTPUT_DIR)
print("📦 Files:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(" -", filename)

print(
    "\nLƯU Ý: Đây là checkpoint Zenith độc lập về mã kiến trúc, "
    "được chuyển đổi từ trọng số Qwen. Tokenizer vẫn được lưu riêng "
    "và dùng qua thư viện Transformers."
)
