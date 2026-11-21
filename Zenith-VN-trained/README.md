---
language:
- vi
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- vietnamese
- zenith-vn
- causal-lm
- text-generation
- conversational
- lora
---

# 🇻🇳 Zenith VN 0.5B

### Vietnamese AI Assistant developed by udVietNamspeedrace

**Zenith VN** is an experimental Vietnamese conversational AI model developed by **udVietNamspeedrace**.

> **Zenith VN — Building Vietnamese AI, one step at a time.**

---

## 🌟 1. Introduction

Zenith VN is designed to:

- Understand and respond to Vietnamese conversations.
- Answer general knowledge questions.
- Assist with technology and programming topics.
- Provide conversational AI experiences.
- Identify itself as Zenith VN.
- Serve as a foundation for future Vietnamese AI development.

This is an experimental project that will continue to improve through future training and development.

## 🧠 2. Model Information

| Property | Details |
|---|---|
| Model Name | Zenith VN |
| Version | 0.5B |
| Model Type | Causal Language Model (`zenith_vn`) |
| Architecture | Qwen2-based Transformer |
| Parameters | Approximately 0.5 billion |
| Main Language | Vietnamese |
| Additional Language | English |
| Training Method | Supervised Fine-Tuning |
| Fine-tuning | LoRA |
| Training Quantization | 4-bit QLoRA |
| Export Format | Safetensors |
| Developer | udVietNamspeedrace |

## 🏗️ 3. Model Development

Zenith VN was developed using the following technologies:

- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- TRL
- Datasets
- Accelerate
- BitsAndBytes

### Training Process

1. Prepare conversational training datasets.
2. Configure the model for training.
3. Apply quantization where appropriate.
4. Configure LoRA adapters.
5. Perform Supervised Fine-Tuning (SFT).
6. Merge LoRA weights into the model when required.
7. Export the resulting model.

## 📦 4. Installation

Install the required libraries:

```bash
pip install torch transformers accelerate
pip install peft huggingface_hub
```

Additional dependencies may be needed depending on the model configuration and hardware.

## 🚀 5. Quick Start

> **Note:** Since `ZenithVNConfig`, `ZenithVNForCausalLM`, and `ZenithVNTokenizer` are custom-registered in the code, please run the registration block before calling `Auto*` classes to load the model.

```python
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
)


class ZenithVNConfig(Qwen2Config):
  model_type = "zenith_vn"


class ZenithVNForCausalLM(Qwen2ForCausalLM):
  config_class = ZenithVNConfig


class ZenithVNTokenizer(Qwen2Tokenizer):
  pass


try:
  AutoConfig.register("zenith_vn", ZenithVNConfig)
  AutoModelForCausalLM.register(ZenithVNConfig, ZenithVNForCausalLM)
  AutoTokenizer.register(ZenithVNConfig, slow_tokenizer_class=ZenithVNTokenizer)
except ValueError:
  pass

model_id = "trangthanhdai/Zenith-0.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, config=config)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
)
model.eval()

messages = [{"role": "user", "content": "Xin chào! Bạn là ai?"}]

if getattr(tokenizer, "chat_template", None):
  inputs = tokenizer.apply_chat_template(
      messages,
      tokenize=True,
      add_generation_prompt=True,
      return_tensors="pt",
  ).to(model.device)
else:
  prompt = "User: Xin chào! Bạn là ai?\nAssistant:"
  inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

with torch.no_grad():
  outputs = model.generate(
      inputs,
      max_new_tokens=150,
      temperature=0.7,
      do_sample=True,
      top_p=0.9,
      pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
      eos_token_id=tokenizer.eos_token_id,
  )

response = tokenizer.decode(
    outputs[0][inputs.shape[-1] :],
    skip_special_tokens=True,
)

print("Zenith VN:", response)
```

## 📁 6. Repository Structure

Expected repository structure:

```text
Zenith-0.5B/
├── README.md
├── config.json
├── generation_config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

The actual files may differ depending on the exported model format or if custom Python modules are added later.

## 💻 7. Hardware Requirements

Recommended hardware depends on the inference precision and sequence length.

- GPU with sufficient VRAM for FP16 inference.
- System RAM for loading and processing model weights.
- CPU inference may also be possible, but can be slower.
- The model has approximately 0.5 billion parameters.

## ⚠️ 8. Limitations

Zenith VN is an experimental AI model.

- It may generate inaccurate information.
- Its reasoning capabilities are limited.
- Its knowledge depends on its training data.
- It may not understand every Vietnamese expression.
- It is not intended to replace professional advice.
- Performance depends on hardware and inference configuration.

## 🔒 9. Security

- Do not upload private access tokens, passwords, personal information, or confidential training data to this repository.
- Always review custom Python model files before executing them.

## 📜 10. License

The license and redistribution conditions for Zenith VN depend on the model components, training materials, datasets, and other resources used during development.

Please verify that you have the necessary rights to distribute the model and its associated files before publishing it.

## 🇻🇳 11. Vision

The goal of Zenith VN is to contribute to the development of Vietnamese conversational AI.

Future development may include:

- Improved Vietnamese language understanding.
- Larger and higher-quality training datasets.
- Better reasoning capabilities.
- Programming assistance.
- More efficient inference.
- Independent model architecture research.

## ❤️ Thank You!

Thank you for visiting and supporting Zenith VN.

Created by **udVietNamspeedrace**

> **Zenith VN — A Vietnamese AI Project. 🇻🇳**