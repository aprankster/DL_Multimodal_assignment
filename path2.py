import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# -----------------------------
# Config
# -----------------------------
IMAGE_PATH = "image.jpg"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"  # Adapter-based VLM

# -----------------------------
# Load image
# -----------------------------
print("[Path2] Loading image...")
image = Image.open(IMAGE_PATH).convert("RGB")

# -----------------------------
# Load model + processor
# -----------------------------
print("[Path2] Loading adapter-based vision-language model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

# -----------------------------
# Run inference
# -----------------------------
prompt = "Describe the image in detail."
inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
).to(model.device)

print("[Path2] Generating description...")
outputs = model.generate(
    **inputs,
    max_new_tokens=120
)

result = processor.decode(outputs[0], skip_special_tokens=True)

print("\n[Path2] Output:")
print(result)
