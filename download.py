from diffusers import (
    HunyuanVideoPipeline, CogVideoXPipeline,
    StableDiffusion3Pipeline, FluxPipeline, WanPipeline,
)


model_path = "THUDM/CogVideoX1.5-5B"

pipe = CogVideoXPipeline.from_pretrained(model_path)

model_path = "hunyuanvideo-community/HunyuanVideo"

pipe = HunyuanVideoPipeline.from_pretrained(model_path)

model_path = "black-forest-labs/FLUX.1-dev"

pipe = FluxPipeline.from_pretrained(model_path)

model_path = "stabilityai/stable-diffusion-3.5-large"

pipe = StableDiffusion3Pipeline.from_pretrained(model_path)

model_path = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

pipe = WanPipeline.from_pretrained(model_path)



print('done')
