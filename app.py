import argparse
import os
from datetime import datetime

import gradio as gr
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding


import gc
from transformers import T5EncoderModel
from diffusers import FluxPipeline, FluxTransformer2DModel

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="booksforcharlie/stable-diffusion-inpainting",  # Change to a copy repo as runawayml delete original repo
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help=(
            "The Path to the checkpoint of trained tryon model."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resource/demo/output",
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--repaint", 
        action="store_true", 
        help="Whether to repaint the result image with the original background."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


args = parse_args()
repo_path = snapshot_download(repo_id=args.resume_path)

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

flush()

ckpt_4bit_id = "sayakpaul/flux.1-dev-nf4-pkg"

text_encoder_2_4bit = T5EncoderModel.from_pretrained(
    ckpt_4bit_id,
    subfolder="text_encoder_2",
)

# image gen pipeline


# Pipeline
pipeline = CatVTONPipeline(
    base_ckpt=args.base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(args.mixed_precision),
    use_tf32=args.allow_tf32,
    device='cuda'
)
# AutoMasker
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda', 
)

def submit_function(
    person_image,
    cloth_image,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type
):
    person_image, mask = person_image["background"], person_image["layers"][0]
    mask = Image.open(mask).convert("L")
    if len(np.unique(np.array(mask))) == 1:
        mask = None
    else:
        mask = np.array(mask)
        mask[mask > 0] = 255
        mask = Image.fromarray(mask)

    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
        os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    generator = None
    if seed != -1:
        generator = torch.Generator(device='cuda').manual_seed(seed)

    person_image = Image.open(person_image).convert("RGB")
    cloth_image = Image.open(cloth_image).convert("RGB")
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))
    
    # Process mask
    if mask is not None:
        mask = resize_and_crop(mask, (args.width, args.height))
    else:
        mask = automasker(
            person_image,
            cloth_type
        )['mask']
    mask = mask_processor.blur(mask, blur_factor=9)

    # Inference
    # try:
    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )[0]
    # except Exception as e:
    #     raise gr.Error(
    #         "An error occurred. Please try again later: {}".format(e)
    #     )
    
    # Post-process
    masked_person = vis_mask(person_image, mask)
    save_result_image = image_grid([person_image, masked_person, cloth_image, result_image], 1, 4)
    save_result_image.save(result_save_path)
    if show_type == "result only":
        return result_image
    else:
        width, height = person_image.size
        if show_type == "input & result":
            condition_width = width // 2
            conditions = image_grid([person_image, cloth_image], 2, 1)
        else:
            condition_width = width // 3
            conditions = image_grid([person_image, masked_person , cloth_image], 3, 1)
        conditions = conditions.resize((condition_width, height), Image.NEAREST)
        new_result_image = Image.new("RGB", (width + condition_width + 5, height))
        new_result_image.paste(conditions, (0, 0))
        new_result_image.paste(result_image, (condition_width + 5, 0))
    return new_result_image


def person_example_fn(image_path):
    return image_path

def generate_person_image(prompt):
    """
    Creates a test image based on the prompt.
    Returns the path to the generated image.
    """

    ckpt_id = "black-forest-labs/FLUX.1-dev"

    image_gen_pipeline = FluxPipeline.from_pretrained(
        ckpt_id,
        text_encoder_2=text_encoder_2_4bit,
        transformer=None,
        vae=None,
        torch_dtype=torch.float16,
    )
    image_gen_pipeline.enable_model_cpu_offload()
    # Create a new image with a random background color

    with torch.no_grad():
        print("Encoding prompts.")
        prompt_embeds, pooled_prompt_embeds, text_ids = image_gen_pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=256
        )

    image_gen_pipeline = image_gen_pipeline.to("cpu")
    del image_gen_pipeline

    flush()

    print(f"prompt_embeds shape: {prompt_embeds.shape}")
    print(f"pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")
    # Add the prompt text to the image
    transformer_4bit = FluxTransformer2DModel.from_pretrained(ckpt_4bit_id, subfolder="transformer")
    image_gen_pipeline = FluxPipeline.from_pretrained(
        ckpt_id,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=transformer_4bit,
        torch_dtype=torch.float16,
    )
    image_gen_pipeline.enable_model_cpu_offload()

    print("Running denoising.")
    height, width = 1024, 1024

    images = image_gen_pipeline(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=50,
        guidance_scale=5.5,
        height=height,
        width=width,
        output_type="pil",
    ).images
    
    # Add current time to make each image unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs('generated_images', exist_ok=True)
    
    # Save the image
    output_path = f'generated_images/generated_{timestamp}.png'
    images[0].save(output_path)
    
    return output_path

HEADER = """
"""

def app_gradio():
    with gr.Blocks(title="CatVTON") as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            with gr.Column(scale=1, min_width=350):
              text_prompt = gr.Textbox(
                label="Describe the person (e.g., 'a young woman in a neutral pose')",
                lines=3
              )

              generate_button = gr.Button("Generate Person Image")
                
                # Hidden image path component
              image_path = gr.Image(
                    type="filepath",
                    interactive=True,
                    visible=False,
                )
                
                # Display generated person image
              person_image = gr.ImageEditor(
                    interactive=True,
                    label="Generated Person Image",
                    type="filepath"
                )

              with gr.Row():
                    with gr.Column(scale=1, min_width=230):
                        cloth_image = gr.Image(
                            interactive=True, label="Condition Image", type="filepath"
                        )
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">Two ways to provide Mask:<br>1. Upload the person image and use the `🖌️` above to draw the Mask (higher priority)<br>2. Select the `Try-On Cloth Type` to generate automatically </span>'
                        )
                        cloth_type = gr.Radio(
                            label="Try-On Cloth Type",
                            choices=["upper", "lower", "overall"],
                            value="upper",
                        )


              submit = gr.Button("Submit")
              gr.Markdown(
                  '<center><span style="color: #FF0000">!!! Click only Once, Wait for Delay !!!</span></center>'
              )
              
              gr.Markdown(
                  '<span style="color: #808080; font-size: small;">Advanced options can adjust details:<br>1. `Inference Step` may enhance details;<br>2. `CFG` is highly correlated with saturation;<br>3. `Random seed` may improve pseudo-shadow.</span>'
              )
              with gr.Accordion("Advanced Options", open=False):
                  num_inference_steps = gr.Slider(
                      label="Inference Step", minimum=10, maximum=100, step=5, value=50
                  )
                  # Guidence Scale
                  guidance_scale = gr.Slider(
                      label="CFG Strenth", minimum=0.0, maximum=7.5, step=0.5, value=2.5
                  )
                  # Random Seed
                  seed = gr.Slider(
                      label="Seed", minimum=-1, maximum=10000, step=1, value=42
                  )
                  show_type = gr.Radio(
                      label="Show Type",
                      choices=["result only", "input & result", "input & mask & result"],
                      value="input & mask & result",
                  )

            with gr.Column(scale=2, min_width=500):
                result_image = gr.Image(interactive=False, label="Result")
                with gr.Row():
                    # Photo Examples
                    root_path = "resource/demo/example"
                    

            image_path.change(
                person_example_fn, inputs=image_path, outputs=person_image
            )

            # Connect the generation button
            generate_button.click(
                generate_person_image,
                inputs=[text_prompt],
                outputs=[person_image]
            )

            submit.click(
                submit_function,
                [
                    person_image,
                    cloth_image,
                    cloth_type,
                    num_inference_steps,
                    guidance_scale,
                    seed,
                    show_type,
                ],
                result_image,
            )
    demo.queue().launch(share=True, show_error=True)


if __name__ == "__main__":
    app_gradio()
