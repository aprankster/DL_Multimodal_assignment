from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

model_id = "llava-hf/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_id)

image = Image.open("image.jpg")
prompt = "Describe this image."

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))
