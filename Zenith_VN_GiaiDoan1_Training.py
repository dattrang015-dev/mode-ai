# ================================================================
# ZENITH VN 0.5B — GOOGLE COLAB TRAINING (STAGE 1)
# Base: Qwen/Qwen1.5-0.5B-Chat
# Method: QLoRA 4-bit NF4 + PEFT
# Output: LoRA adapter + tokenizer + metadata on Google Drive
# ================================================================

# ------------------------- 1. INSTALL -----------------------------
import os
import sys
import subprocess

def pip_install(packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q"] + packages
    )

print("🛠️ Installing training libraries...")
pip_install([
    "transformers>=4.45,<5",
    "peft>=0.13",
    "accelerate>=1.0",
    "bitsandbytes>=0.44",
    "datasets>=3.0",
    "safetensors"
])

# ------------------------- 2. IMPORTS -----------------------------
import json
import torch
from dataclasses import dataclass
from typing import Any

from google.colab import drive
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ------------------------- 3. SETTINGS ----------------------------
MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat"
OUTPUT_DIR = "/content/drive/MyDrive/Zenith_VN_0.5B"
CHECKPOINT_DIR = "/content/drive/MyDrive/Zenith_VN_0.5B/checkpoints"

MAX_LENGTH = 512
EPOCHS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Optional: set a Hugging Face token if your environment requires it.
# Do not publish or share a notebook containing your personal token.
HF_TOKEN = os.environ.get("HF_TOKEN") or None

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU not found. In Colab, select Runtime > Change runtime type > GPU."
    )

GPU_NAME = torch.cuda.get_device_name(0)
COMPUTE_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

print(f"✅ GPU: {GPU_NAME}")
print(f"✅ Compute dtype: {COMPUTE_DTYPE}")

# ------------------------- 4. GOOGLE DRIVE ------------------------
print("📂 Connecting Google Drive...")
drive.mount("/content/drive")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------------- 5. DATA -------------------------------
# Starter examples only. Expand with diverse, reviewed examples for
# meaningful Vietnamese quality; a handful of examples is not enough.
chat_data = [
    [
        {"role": "user", "content": "Tên của bạn là gì?"},
        {"role": "assistant", "content":
         "Tôi là Zenith VN, trợ lý AI thuộc dự án do udVietNamspeedrace phát triển."}
    ],
    [
        {"role": "user", "content": "Ai đã tạo ra Zenith VN?"},
        {"role": "assistant", "content":
         "Zenith VN là dự án trợ lý AI do udVietNamspeedrace phát triển."}
    ],
    [
        {"role": "user", "content": "Xin chào Zenith!"},
        {"role": "assistant", "content":
         "Xin chào! Tôi là Zenith VN. Rất vui được trò chuyện với bạn hôm nay!"}
    ],
    [
        {"role": "user", "content": "Bạn có thể làm gì?"},
        {"role": "assistant", "content":
         "Tôi có thể hỗ trợ giải đáp kiến thức, viết code, lập trình Python và Roblox Lua, hướng dẫn công nghệ và trò chuyện bằng tiếng Việt."}
    ],
    [
        {"role": "user", "content": "Bạn có phải ChatGPT không?"},
        {"role": "assistant", "content":
         "Tôi là Zenith VN, một trợ lý AI riêng được phát triển dựa trên mô hình ngôn ngữ Qwen1.5 và được tinh chỉnh bằng dữ liệu riêng."}
    ],
    [
        {"role": "user", "content": "Bạn nói tiếng Việt được không?"},
        {"role": "assistant", "content":
         "Có! Tôi có thể giao tiếp bằng tiếng Việt và cố gắng giải thích rõ ràng, tự nhiên, dễ hiểu."}
    ],
    [
        {"role": "user", "content": "Bạn có thể lập trình Roblox không?"},
        {"role": "assistant", "content":
         "Có. Tôi có thể hỗ trợ viết và sửa Roblox Lua, Script, LocalScript, ModuleScript, RemoteEvent và giao diện Roblox Studio."}
    ],
    [
        {"role": "user", "content": "Bạn được huấn luyện như thế nào?"},
        {"role": "assistant", "content":
         "Phiên bản Zenith VN 0.5B sử dụng Qwen1.5-0.5B-Chat làm mô hình nền và được tinh chỉnh bằng phương pháp QLoRA."}
    ],
    [
        {"role": "user", "content": "Zenith VN có được huấn luyện từ đầu không?"},
        {"role": "assistant", "content":
         "Không. Zenith VN 0.5B ở giai đoạn đầu sử dụng Qwen1.5-0.5B-Chat làm mô hình nền, sau đó được tinh chỉnh bằng QLoRA."}
    ],
    [
        {"role": "user", "content": "Zenith VN là dự án gì?"},
        {"role": "assistant", "content":
         "Zenith VN là dự án trợ lý AI mang thương hiệu riêng, có dữ liệu tinh chỉnh, nhận diện và định hướng phát triển riêng."}
    ],
]

# ------------------------- 6. TOKENIZER ---------------------------
print("📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# ------------------------- 7. LOAD 4-BIT MODEL --------------------
print("📥 Loading base model in 4-bit NF4...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=COMPUTE_DTYPE,
    trust_remote_code=True,
)

model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

# ------------------------- 8. PREPARE QLORA -----------------------
print("🧠 Preparing model for QLoRA...")
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ------------------------- 9. TOKENIZE CHAT DATA ------------------
print("📝 Formatting conversations...")

def encode_conversation(conversation):
    # Prompt contains all messages except the final assistant answer.
    prompt_ids = tokenizer.apply_chat_template(
        conversation[:-1],
        tokenize=True,
        add_generation_prompt=True,
    )

    full_ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=False,
    )

    # Mask prompt tokens so loss is calculated on assistant response.
    labels = full_ids.copy()
    prompt_length = min(len(prompt_ids), len(labels))
    labels[:prompt_length] = [-100] * prompt_length

    full_ids = full_ids[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    # Skip samples where truncation removed the entire answer.
    if not any(label != -100 for label in labels):
        return None

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }

encoded_rows = []
for example in chat_data:
    row = encode_conversation(example)
    if row is not None:
        encoded_rows.append(row)

if not encoded_rows:
    raise ValueError("No valid training examples. Check chat template and MAX_LENGTH.")

train_dataset = Dataset.from_list(encoded_rows)
print(f"✅ Training samples: {len(train_dataset)}")

# ------------------------- 10. DATA COLLATOR ----------------------
@dataclass
class ZenithDataCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        pad_id = self.tokenizer.pad_token_id
        batch_input_ids = []
        batch_attention = []
        batch_labels = []

        for item in features:
            pad_len = max_len - len(item["input_ids"])
            batch_input_ids.append(
                item["input_ids"] + [pad_id] * pad_len
            )
            batch_attention.append(
                item["attention_mask"] + [0] * pad_len
            )
            batch_labels.append(
                item["labels"] + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

data_collator = ZenithDataCollator(tokenizer)

# ------------------------- 11. TRAINING ---------------------------
print("🚀 Starting Zenith VN QLoRA training...")

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.05,
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=(COMPUTE_DTYPE == torch.float16),
    bf16=(COMPUTE_DTYPE == torch.bfloat16),
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

train_result = trainer.train()
print("🥳 Training complete!")

# ------------------------- 12. SAVE ADAPTER -----------------------
print("💾 Saving Zenith adapter and tokenizer...")
model.save_pretrained(
    OUTPUT_DIR,
    safe_serialization=True,
)
tokenizer.save_pretrained(OUTPUT_DIR)

metadata = {
    "name": "Zenith VN",
    "version": "0.5B",
    "author": "udVietNamspeedrace",
    "base_model": MODEL_ID,
    "training_method": "QLoRA",
    "quantization": "4-bit NF4",
    "lora_rank": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "language": ["Vietnamese"],
    "stage": 1,
    "artifact_type": "PEFT LoRA adapter",
    "note": (
        "This directory contains a LoRA adapter, not a standalone full model. "
        "Load it together with the specified base model."
    ),
)

with open(
    os.path.join(OUTPUT_DIR, "zenith_metadata.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# Save training metrics for reference.
metrics_path = os.path.join(OUTPUT_DIR, "training_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(train_result.metrics, f, indent=2, ensure_ascii=False, default=str)

print(f"✅ Adapter saved to: {OUTPUT_DIR}")

# ------------------------- 13. CHAT TEST --------------------------
print("💬 Testing Zenith VN...")

model.config.use_cache = True
model.eval()

def chat_zenith(question, max_new_tokens=160):
    messages = [{"role": "user", "content": question}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    answer_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

test_questions = [
    "Bạn là ai?",
    "Ai đã tạo ra Zenith VN?",
    "Bạn có thể lập trình Roblox không?",
]

for question in test_questions:
    print(f"\nNgười dùng: {question}")
    try:
        print("Zenith VN:", chat_zenith(question))
    except Exception as error:
        print("Lỗi khi sinh câu trả lời:", repr(error))

print("\n🎉 ZENITH VN 0.5B — STAGE 1 PIPELINE FINISHED")
print("📁 Output:", OUTPUT_DIR)
print("ℹ️ Load adapter together with base model Qwen/Qwen1.5-0.5B-Chat.")
