from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

model_id = "Salesforce/blip2-flan-t5-base"

processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda"
)

image = Image.open("image.jpg")
prompt = "Explain the scene."

inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))
