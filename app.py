import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import io

# ===== Premium Keys =====
premium_keys = {"PREMIUM123", "VIP456"}  # Add/remove keys here

# ===== Load Model =====
model_id = "stabilityai/sd-turbo"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# ===== Watermark Function =====
def add_watermark(image, text="I m hckr"):
    draw = ImageDraw.Draw(image)
    font_size = max(20, image.width // 20)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    text_width, text_height = draw.textsize(text, font=font)
    x = image.width - text_width - 10
    y = image.height - text_height - 10
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 128))
    return image

# ===== Image Generation =====
def generate(prompt, negative_prompt, strength, style, key):
    is_premium = key in premium_keys
    steps = 4
    size = (640, 640) if is_premium else (448, 448)
    num_images = 4 if is_premium else 2

    # Apply style
    if style != "None":
        prompt = f"{prompt}, {style} style"

    images = []
    for _ in range(num_images):
        img = pipe(
            prompt,
            negative_prompt=negative_prompt,
            guidance_scale=strength,
            num_inference_steps=steps
        ).images[0]
        img = img.resize(size)
        if not is_premium:
            img = add_watermark(img)
        images.append(img)
    return images

# ===== Styles =====
styles = ["None", "Cyberpunk", "Fantasy Art", "Photorealistic", "Anime", "Pixel Art", "Watercolor"]

# ===== Gradio Interface =====
with gr.Blocks(css="style.css") as demo:
    gr.Markdown("# 🖼 AI Image Generator with Premium & Watermark System")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt")
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Things to avoid")
            strength = gr.Slider(1, 10, value=7, step=0.5, label="Prompt Strength")
            style = gr.Dropdown(choices=styles, value="None", label="Style")
            key = gr.Textbox(label="Premium Key (Leave blank if free user)")
            generate_btn = gr.Button("Generate Image")
        with gr.Column():
            output = gr.Gallery(label="Generated Images").style(grid=2)

    generate_btn.click(
        fn=generate,
        inputs=[prompt, negative_prompt, strength, style, key],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
