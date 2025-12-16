import whisper
import torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline

# -------- 1. Speech to Text --------
whisper_model = whisper.load_model("small")
result = whisper_model.transcribe("audio.wav")
text_prompt = result["text"]
print("Transcribed:", text_prompt)

# -------- 2. LLM Reasoning (LIGHT MODEL) --------
model_id = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda"
)

inputs = tokenizer(text_prompt, return_tensors="pt").to("cuda")
outputs = llm.generate(**inputs, max_new_tokens=80)
final_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Final prompt:", final_prompt)

# Free GPU memory
del llm, inputs, outputs
torch.cuda.empty_cache()
gc.collect()

# -------- 3. Text to Image --------
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

image = pipe(final_prompt, num_inference_steps=25).images[0]
image.save("output.png")

print(" output.png saved")
