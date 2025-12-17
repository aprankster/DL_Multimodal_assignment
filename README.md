# Multimodal Deep Learning Project

## Overview

This project demonstrates **three distinct multimodal deep learning architectures** for combining language, vision, and speech. Each architecture is implemented as a **standalone runnable script** and executed locally using an NVIDIA RTX 4060 GPU on WSL2.


---

## Project Structure

```
DL/
├── mmproj/                 # Python virtual environment
├── audio.wav               # Input audio for Path 1
├── image.jpg               # Input image for Path 2 and Path 3
├── path1.py                # Path 1: LLM + External Tools
├── path2.py                # Path 2: LLM + Adapters (LLaVA-style)
├── path3.py                # Path 3: Unified Multimodal Model (BLIP-2)
```

---

## Paths Implemented

### Path 1 — LLM + External Tools

**Pipeline:**

1. Speech-to-text using Whisper
2. Prompt refinement using an LLM
3. Image generation using Stable Diffusion

**Input:** `audio.wav`

**Output:** `output.png`

**Run:**

```bash
python path1.py
```

---

### Path 2 — LLM + Adapters (Vision-Language)

**Pipeline:**

* Image is encoded using a vision encoder

* Visual features are aligned with a language model via adapter layers

* The language model generates a detailed textual description of the image

**Model** `llava-hf/llava-1.5-7b-hf`

**Input:** `image.jpg`

**Output:** Textual description printed to terminal

**Run:**

```bash
python path2.py
```

**Characteristics:**

* Explicit multimodal alignment

* Strong image understanding

* More efficient than tool-based pipelines
---

### Path 3 — Unified Multimodal Model

**Pipeline:**

* A single pretrained model jointly processes visual and textual inputs
* Vision and language reasoning occur within one unified architecture

**Model** `Salesforce/blip2-flan-t5-base`

**Input:** `image.jpg`

**Output:** Natural language explanation of the image

**Run:**

```bash
python path3.py
```

> **Note:** This path uses an inference-only pretrained model. No training or fine-tuning is performed.

---

## Setup Instructions

### 1. Create Virtual Environment

```bash
python3 -m venv mmproj
source mmproj/bin/activate
```

### 2. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers accelerate pillow openai-whisper
sudo apt install ffmpeg
```

### 3. Hugging Face Authentication (Required for Path 3)

```bash
huggingface-cli login
```

Then accept the BLIP-2 license at:
[https://huggingface.co/Salesforce/blip2-flan-t5-base](https://huggingface.co/Salesforce/blip2-flan-t5-base)

---

## Hardware Used

* GPU: NVIDIA RTX 4060 (8GB VRAM)
* OS: WSL2 (Ubuntu)

---

## Key Observations

* Path 1 is modular but computationally heavy
* Path 2 balances flexibility and efficiency
* Path 3 is simplest but least adaptable

---

## Key Observations
| Path   | Architecture             | Strengths                   | Limitations      |
| ------ | ------------------------ | --------------------------- | ---------------- |
| Path 1 | Tool-based pipeline      | Modular, interpretable      | Higher latency   |
| Path 2 | Adapter-based VLM        | Strong multimodal alignment | Large model size |
| Path 3 | Unified multimodal model | Simple end-to-end reasoning | Less modular     |

---

## Conclusion

This project demonstrates how different multimodal architectures trade off modularity, performance, and integration.
By comparing tool-based pipelines, adapter-based vision–language models, and unified multimodal transformers, it highlights key design choices in modern multimodal systems.
