import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# -----------------------------
# Config
# -----------------------------
IMAGE_PATH = "image.jpg"
MODEL_ID = "Salesforce/blip2-flan-t5-base"

# -----------------------------
# Load image
# -----------------------------
print("[Path3] Loading image...")
image = Image.open(IMAGE_PATH).convert("RGB")

# -----------------------------
# Load unified multimodal model
# -----------------------------
print("[Path3] Loading unified multimodal model...")
processor = Blip2Processor.from_pretrained(MODEL_ID)

model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

# -----------------------------
# Inference
# -----------------------------
prompt = "Explain what is happening in this image."
inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt"
).to(model.device)

print("[Path3] Generating response...")
outputs = model.generate(
    **inputs,
    max_new_tokens=100
)

result = processor.decode(outputs[0], skip_special_tokens=True)

print("\n[Path3] Output:")
print(result)
