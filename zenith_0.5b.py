# =====================================================================
# BƯỚC 1: CÀI ĐẶT THƯ VIỆN ĐỒNG BỘ PHẦN CỨNG
# =====================================================================
print("🛠️ Đang cấu hình lại các thư viện để đồng bộ phần cứng...")
!pip install "setuptools<82" -q
!pip install transformers trl peft bitsandbytes datasets accelerate -q

import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig
from trl import SFTTrainer

# =====================================================================
# BƯỚC 2: CẤU HÌNH THÔNG TIN TÀI KHOẢN (BẢO MẬT NỘI BỘ)
# =====================================================================
HF_TOKEN = "hf của bạn"
os.environ["HF_TOKEN"] = HF_TOKEN

# =====================================================================
# BƯỚC 3: TẢI MÔ HÌNH QWEN 1.5 VÀ ÉP TOÀN BỘ CẤU TRÚC VỀ FLOAT16
# =====================================================================
model_id = "Qwen/Qwen1.5-0.5B-Chat"
print(f"\n📥 Đang tải bộ giải mã (Tokenizer) của {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"📥 Đang tải cấu trúc nơ-ron {model_id} dưới dạng nén 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, # Đảm bảo tính toán lượng hóa bằng float16
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN,
    torch_dtype=torch.float16 # Ép cấu trúc ma trận gốc về hẳn Float16
)

# Thao tác ép tất cả các lớp nơ-ron còn sót lại (như LayerNorm hay Embedding) về float16 để triệt tiêu hoàn toàn BFloat16
for name, param in model.named_parameters():
    if "norm" in name or "embed" in name:
        param.data = param.data.to(torch.float16)

# =====================================================================
# BƯỚC 4: TỰ ĐỘNG MÃ HÓA VÀ ĐỊNH DẠNG DỮ LIỆU SẠCH THEO CHAT TEMPLATE
# =====================================================================
print("\n📝 Chuẩn hóa dữ liệu hội thoại theo chuẩn Chat Template của Qwen...")

# Định dạng lại dữ liệu dưới dạng cấu trúc hội thoại chuẩn để khớp 100% với Chatbot
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

# Chuyển đổi dữ liệu hội thoại sang dạng Text thô bằng Chat Template
formatted_texts = []
for messages in chat_data:
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    formatted_texts.append(text)

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
# BƯỚC 6: HUẦN LUYỆN CHUẨN SẠCH VÀ TĂNG SỐ EPOCHS
# =====================================================================
print("\n🚀 Kích hoạt quy trình SFT Trainer...")

training_args = TrainingArguments(
    output_dir="./results_qwen_local",
    num_train_epochs=15,          # Giữ nguyên 15 epochs để học thuộc các thông tin mới
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

# Chạy huấn luyện chính thức
trainer.train()

print("\n🥳 Huấn luyện thành công vẹn toàn!")

# =====================================================================
# BƯỚC 7: ĐÓNG GÓI, CẬP NHẬT METADATA VÀ LƯU MÔ HÌNH
# =====================================================================
import json

output_dir = "./results_qwen_local"

# Lưu model và tokenizer như bình thường
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Cập nhật trực tiếp tệp config.json để gắn nhãn định danh tác giả và tên thương hiệu
config_path = os.path.join(output_dir, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    # Bổ sung thông tin định danh cá nhân và thương hiệu vào metadata của model
    config_data["architectures"] = ["Qwen2ForCausalLM"]
    config_data["model_type"] = "qwen2"
    config_data["_name_or_path"] = "Zenith-VN-0.5B"
    config_data["author"] = "udVietNamspeedrace"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print("🏷️ Đã cập nhật thành công metadata định danh vào config.json!")

print(f"\n🎉 [HOÀN TẤT] Bộ não Zenith đã được đóng gói và lưu tại {output_dir}!")
