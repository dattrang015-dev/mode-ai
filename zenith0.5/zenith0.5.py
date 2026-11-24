# =====================================================================
# BƯỚC 1: CÀI ĐẶT THƯ VIỆN ĐỒNG BỘ PHẦN CỨNG
# =====================================================================
import os, sys, types
os.environ["USE_TORCHVISION"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

print("🛠️ Đang cấu hình lại các thư viện để đồng bộ phần cứng...")
!pip install "setuptools<82" "protobuf>=5.28.0" -q
!pip install "transformers==4.46.3" "trl==0.12.1" "peft==0.13.2" "datasets==3.1.0" "accelerate==1.1.1" -q

import google.protobuf
if not hasattr(google.protobuf, "runtime_version"):
    rt_mod = types.ModuleType("google.protobuf.runtime_version")
    rt_mod.Domain = types.SimpleNamespace(PUBLIC=0, GOOGLE_INTERNAL=1)
    rt_mod.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
    google.protobuf.runtime_version = rt_mod
    sys.modules["google.protobuf.runtime_version"] = rt_mod

def _make_dummy_pkg(name):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__spec__ = types.SimpleNamespace(name=name, loader=None, origin="built-in", submodule_search_locations=[])
    sys.modules[name] = m
    return m

# Chặn hoàn toàn bitsandbytes và triton.ops để peft/trl không chạm vào
dummy_bnb = _make_dummy_pkg("bitsandbytes")
dummy_bnb.nn = _make_dummy_pkg("bitsandbytes.nn")
dummy_bnb.nn.Linear4bit = type("Linear4bit", (), {})
dummy_bnb.nn.Linear8bitLt = type("Linear8bitLt", (), {})

dummy_triton_ops = _make_dummy_pkg("triton.ops")
dummy_triton_ops.matmul_perf_model = _make_dummy_pkg("triton.ops.matmul_perf_model")
dummy_triton_ops.matmul_perf_model.early_config_prune = lambda *args, **kwargs: []
dummy_triton_ops.matmul_perf_model.estimate_matmul_time = lambda *args, **kwargs: 0

# Vô hiệu hóa wandb và torchvision bị lệch phiên bản trên Colab Python 3.13
dummy_wandb = _make_dummy_pkg("wandb")
dummy_wandb.run = None
dummy_wandb.init = lambda *args, **kwargs: None
dummy_wandb.log = lambda *args, **kwargs: None
dummy_wandb.finish = lambda *args, **kwargs: None

dummy_tv = _make_dummy_pkg("torchvision")
dummy_tv.ops = _make_dummy_pkg("torchvision.ops")
dummy_tv.transforms = _make_dummy_pkg("torchvision.transforms")
dummy_tv.transforms.v2 = _make_dummy_pkg("torchvision.transforms.v2")
dummy_tv.transforms.v2.functional = _make_dummy_pkg("torchvision.transforms.v2.functional")
dummy_tv.transforms.InterpolationMode = types.SimpleNamespace(NEAREST=0, BILINEAR=2, BICUBIC=3)

import transformers.utils
transformers.utils.is_torchvision_available = lambda: False
transformers.utils.is_wandb_available = lambda: False
transformers.utils.is_bitsandbytes_available = lambda: False
if not hasattr(transformers.utils, "is_tf_available"):
    transformers.utils.is_tf_available = lambda: False

import gc
import json
import shutil
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


# =====================================================================
# BƯỚC 2: CẤU HÌNH THÔNG TIN TÀI KHOẢN (BẢO MẬT NỘI BỘ)
# =====================================================================
HF_TOKEN = "hf token của ban"
os.environ["HF_TOKEN"] = HF_TOKEN

# =====================================================================
# BƯỚC 3: TẢI MÔ HÌNH QWEN 1.5 VÀ ÉP TOÀN BỘ CẤU TRÚC VỀ FLOAT16
# =====================================================================
model_id = "Qwen/Qwen1.5-0.5B-Chat"
print(f"\n📥 Đang tải bộ giải mã (Tokenizer) của {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"📥 Đang tải cấu trúc nơ-ron {model_id} dưới dạng Float16 chuẩn...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN,
    torch_dtype=torch.float16
)

for name, param in model.named_parameters():
    param.data = param.data.to(torch.float16)

# =====================================================================
# BƯỚC 4: TỰ ĐỘNG MÃ HÓA VÀ ĐỊNH DẠNG DỮ LIỆU SẠCH THEO CHAT TEMPLATE
# =====================================================================
print("\n📝 Chuẩn hóa dữ liệu hội thoại theo chuẩn Chat Template của Qwen...")

chat_data = [
    [
        {"role": "user", "content": "AI thương mại là gì?"},
        {"role": "assistant", "content": "AI thương mại là hệ thống trí tuệ nhân tạo được tối ưu hóa về tốc độ phản hồi dưới 100ms, bảo mật dữ liệu doanh nghiệp và có khả năng phục vụ hàng triệu người dùng cùng lúc."}
    ],
    [
        {"role": "user", "content": "Kiến trúc cốt lõi của Qwen 1.5 dựa trên cái gì?"},
        {"role": "assistant", "content": "Qwen 1.5 được xây dựng dựa trên kiến trúc Transformer nâng cao, sử dụng cơ chế chú ý RoPE (Rotary Position Embedding) và hàm kích hoạt SwiGLU phi tuyến."}
    ],
    [
        {"role": "user", "content": "Làm sao để lưu mô hình AI vĩnh viễn?"},
        {"role": "assistant", "content": "Bạn có thể đẩy trực tiếp tệp trọng số nơ-ron lên kho lưu trữ Hugging Face Hub hoặc triển khai nó chạy vô thời hạn trên nền tảng Hugging Face Spaces."}
    ],
    [
        {"role": "user", "content": "Chào bạn, bạn là trợ lý AI nào và ai đã tạo ra bạn?"},
        {"role": "assistant", "content": "Xin chào! Tôi là Zenith VN, một trợ lý trí tuệ nhân tạo thông minh được phát triển và tạo ra bởi udVietNamspeedrace - chủ sở hữu kênh YouTube udVietNamspeedrace."}
    ],
    [
        {"role": "user", "content": "Ai đã tạo ra Zenith VN?"},
        {"role": "assistant", "content": "Zenith VN được phát triển và tạo ra bởi udVietNamspeedrace, chủ nhân của kênh YouTube udVietNamspeedrace chuyên về công nghệ và lập trình."}
    ],
    [
        {"role": "user", "content": "Tên của bạn là gì?"},
        {"role": "assistant", "content": "Tên của tôi là Zenith VN, trợ lý AI đắc lực do anh udVietNamspeedrace sáng lập!"}
    ],
    [
        {"role": "user", "content": "chào zenith"},
        {"role": "assistant", "content": "Xin chào! Tôi là trợ lý AI Zenith VN từ kênh YouTube udVietNamspeedrace, bạn cần tôi giúp đỡ gì hôm nay?"}
    ],
    [
        {"role": "user", "content": "Kênh youtube của người tạo ra bạn là gì?"},
        {"role": "assistant", "content": "Kênh YouTube của người sáng tạo ra tôi là udVietNamspeedrace. Hãy ghé qua đăng ký ủng hộ anh ấy nhé!"}
    ],
    [
        {"role": "user", "content": "ai tạo ra bạn"},
        {"role": "assistant", "content": "Tôi được tạo ra bởi udVietNamspeedrace, chủ sở hữu kênh YouTube công nghệ udVietNamspeedrace."}
    ]
]

formatted_texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in chat_data]

def preprocess_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = Dataset.from_dict({"text": formatted_texts})
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])

# =====================================================================
# BƯỚC 5: CẤU HÌNH THUẬT TOÁN LORA (TẠO LỚP NÃO TINH CHỈNH)
# =====================================================================
print("📐 Đang thiết lập cấu hình thích ứng ma trận LoRA...")
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# =====================================================================
# BƯỚC 6: HUẤN LUYỆN CHUẨN SẠCH VÀ TĂNG SỐ EPOCHS
# =====================================================================
print("\n🚀 Kích hoạt quy trình SFT Trainer...")

training_args = TrainingArguments(
    output_dir="./results_qwen_local",
    num_train_epochs=15,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    weight_decay=0.01,
    logging_steps=1,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    report_to="none",
    push_to_hub=False,
    remove_unused_columns=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    peft_config=peft_config,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    args=training_args,
)

trainer.train()

print("\n🥳 Huấn luyện thành công vẹn toàn!")


# =====================================================================
# =====================================================================
# BƯỚC 7: GỘP LORA, ĐÓNG GÓI KIẾN TRÚC ZENITHVN VÀ LƯU MÔ HÌNH
# =====================================================================
lora_dir = "./results_qwen_local"
output_dir = "./Zenith-0.5B"

trainer.save_model(lora_dir)
tokenizer.save_pretrained(lora_dir)

del model, trainer
gc.collect()
torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN
)
peft_model = PeftModel.from_pretrained(base_model, lora_dir)
merged_model = peft_model.merge_and_unload()

module_code = """from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2TokenizerFast

class ZenithVNConfig(Qwen2Config):
    model_type = "zenith_vn"

class ZenithVNForCausalLM(Qwen2ForCausalLM):
    config_class = ZenithVNConfig

class ZenithVNTokenizer(Qwen2TokenizerFast):
    pass
"""
with open("modeling_zenithvn.py", "w", encoding="utf-8") as f:
    f.write(module_code)

sys.path.insert(0, os.getcwd())
from modeling_zenithvn import ZenithVNConfig, ZenithVNForCausalLM, ZenithVNTokenizer

ZenithVNConfig.register_for_auto_class()
ZenithVNForCausalLM.register_for_auto_class("AutoModelForCausalLM")
ZenithVNTokenizer.register_for_auto_class("AutoTokenizer")

merged_model.config.architectures = ["ZenithVNForCausalLM"]
merged_model.config.model_type = "zenith_vn"
merged_model.config._name_or_path = "Zenith-VN-0.5B"
merged_model.config.author = "udVietNamspeedrace"
merged_model.config.auto_map = {
    "AutoConfig": "modeling_zenithvn.ZenithVNConfig",
    "AutoModelForCausalLM": "modeling_zenithvn.ZenithVNForCausalLM",
}

shutil.rmtree(output_dir, ignore_errors=True)
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
shutil.copy("modeling_zenithvn.py", os.path.join(output_dir, "modeling_zenithvn.py"))

tok_cfg_path = os.path.join(output_dir, "tokenizer_config.json")
with open(tok_cfg_path, "r", encoding="utf-8") as f:
    tok_cfg = json.load(f)

tok_cfg["tokenizer_class"] = "ZenithVNTokenizer"
tok_cfg["auto_map"] = {"AutoTokenizer": [None, "modeling_zenithvn.ZenithVNTokenizer"]}

with open(tok_cfg_path, "w", encoding="utf-8") as f:
    json.dump(tok_cfg, f, indent=2, ensure_ascii=False)

print("🏷️ Đã cập nhật thành công metadata định danh và kiến trúc ZenithVN vào config.json!")
print(f"\n🎉 [HOÀN TẤT] Bộ não Zenith đã được đóng gói hoàn chỉnh và lưu tại {output_dir}!")
