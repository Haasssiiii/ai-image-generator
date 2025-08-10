import gradio as gr
import json
import os
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageDraw, ImageFont

# ---------- CONFIG ----------
ADMIN_PASSWORD = "adminpass123"  # Change this!
PREMIUM_KEYS_FILE = "premium_keys.json"

# ---------- LOAD / SAVE PREMIUM KEYS ----------
def load_premium_keys():
    if os.path.exists(PREMIUM_KEYS_FILE):
        with open(PREMIUM_KEYS_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_premium_keys(keys):
    with open(PREMIUM_KEYS_FILE, "w") as f:
        json.dump(list(keys), f)

premium_keys = load_premium_keys()

# ---------- LOAD MODEL ----------
model_id = "stabilityai/sd-turbo"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")

# ---------- ADD WATERMARK ----------
def add_watermark(img, text="I m hckr"):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    w, h = img.size
    text_w, text_h = draw.textsize(text, font)
    draw.text((w - text_w - 10, h - text_h - 10), text, font=font, fill=(255, 0, 0))
    return img

# ---------- IMAGE GENERATION ----------
def generate_image(prompt, negative_prompt, style, prompt_strength, num_images, key):
    is_premium = key in premium_keys

    # Apply style
    if style != "None":
        prompt = f"{prompt}, {style}"

    # Set resolution and image count
    if is_premium:
        width, height = 640, 640
        images_to_generate = 4
    else:
        width, height = 448, 448
        images_to_generate = 2

    num_images = min(num_images, images_to_generate)

    images = []
    for _ in range(num_images):
        img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=prompt_strength,
            num_inference_steps=4,
            width=width,
            height=height
        ).images[0]

        if not is_premium:
            img = add_watermark(img)

        images.append(img)
    return images

# ---------- ADMIN FUNCTIONS ----------
def admin_login(password):
    if password == ADMIN_PASSWORD:
        return gr.update(visible=True), ""
    else:
        return gr.update(visible=False), "Incorrect password"

def add_key(new_key):
    premium_keys.add(new_key)
    save_premium_keys(premium_keys)
    return list(premium_keys), f"Key '{new_key}' added."

def remove_key(key_to_remove):
    if key_to_remove in premium_keys:
        premium_keys.remove(key_to_remove)
        save_premium_keys(premium_keys)
        return list(premium_keys), f"Key '{key_to_remove}' removed."
    else:
        return list(premium_keys), "Key not found."

# ---------- UI ----------
with gr.Blocks(css="style.css") as demo:
    gr.Markdown("# 🖌 AI Image Generator with Premium Access")
    with gr.Tab("Generate Images"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt")
                negative_prompt = gr.Textbox(label="Negative Prompt")
                style = gr.Dropdown(
                    ["None", "Cyberpunk", "Anime", "Photorealistic", "Fantasy", "Pixel Art", "Watercolor"],
                    value="None",
                    label="Style"
                )
                prompt_strength = gr.Slider(1, 10, value=7, step=1, label="Prompt Strength")
                num_images = gr.Slider(1, 4, value=2, step=1, label="Number of Images")
                key = gr.Textbox(label="Premium Key (leave empty if free user)")
                generate_btn = gr.Button("Generate")
            with gr.Column():
                gallery = gr.Gallery(label="Generated Images").style(grid=2)

        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, negative_prompt, style, prompt_strength, num_images, key],
            outputs=gallery
        )

    with gr.Tab("Admin Panel"):
        admin_password = gr.Textbox(label="Admin Password", type="password")
        login_btn = gr.Button("Login")
        with gr.Group(visible=False) as admin_group:
            new_key = gr.Textbox(label="Add Premium Key")
            add_btn = gr.Button("Add Key")
            remove_key_input = gr.Textbox(label="Remove Premium Key")
            remove_btn = gr.Button("Remove Key")
            keys_list = gr.JSON(label="Current Premium Keys")
            add_btn.click(add_key, inputs=new_key, outputs=[keys_list, gr.Textbox()])
            remove_btn.click(remove_key, inputs=remove_key_input, outputs=[keys_list, gr.Textbox()])

        login_btn.click(admin_login, inputs=admin_password, outputs=[admin_group, gr.Textbox()])

demo.launch()
