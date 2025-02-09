import argparse
import os
from datetime import datetime
from openai import AzureOpenAI
import gradio as gr
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image
import base64
from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding
from dotenv import load_dotenv
from io import BytesIO


load_dotenv()

import gc
from transformers import T5EncoderModel
from diffusers import FluxPipeline, FluxTransformer2DModel

openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_version="2024-02-15-preview",
            azure_deployment="gpt-4o-mvp-dev"
)

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
    show_type,
    campaign_context,
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
    # Generate product description
    product_description = generate_upper_cloth_description(cloth_image, cloth_type)
    
    # Generate captions for the campaign
    captions = generate_captions(product_description, campaign_context)

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
    return new_result_image, captions


def person_example_fn(image_path):
    return image_path

def generate_person_image(text, cloth_description):
    """
    Creates a test image based on the prompt.
    Returns the path to the generated image.
    """
    prompt = generate_ai_model_prompt(text, cloth_description)
    ckpt_id = "black-forest-labs/FLUX.1-dev"

    print("generating image with prompt: ", prompt)
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

def pil_image_to_base64(image, format: str = "PNG") -> str:
    """
    Converts an image to a Base64 encoded string.

    Args:
        image: Either a file path (str) or a PIL Image object
        format (str): The format to save the image as (default is PNG).

    Returns:
        str: A Base64 encoded string of the image.
    """
    try:
        # If image is a file path, open it
        if isinstance(image, str):
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be either a file path or a PIL Image object")
        
        # Convert the image to Base64
        buffered = BytesIO()
        image.save(buffered, format=format)
        buffered.seek(0)  # Go to the start of the BytesIO stream
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return image_base64
    except Exception as e:
        print(f"Error converting image to Base64: {e}")
        raise e

def generate_upper_cloth_description(product_image, cloth_type: str):
    try:
        base_64_image = pil_image_to_base64(product_image)

        if cloth_type == "upper":
            system_prompt = """
                You are world class fahsion designer
                Your task is to Write a detailed description of the upper body garment shown in the image, focusing on its fit, sleeve style, fabric type, neckline, and any notable design elements or features in one or two lines for given image.
                Don't start with "This image shows a pair of beige cargo ..." but instead start with "a pair of beige cargo ..."
            """
        elif cloth_type == "lower":
            system_prompt = """
                You are world class fahsion designer
                Your task is to Write a detailed description of the lower body garment shown in the image, focusing on its fit, fabric type, waist style, and any notable design elements or features in one or two lines for given image.
                Don't start with "This image shows a pair of beige cargo ..." but instead start with "a pair of beige cargo ..."
            """
        elif cloth_type == "overall":
            system_prompt = """
                You are world class fahsion designer
                Your task is to Write a detailed description of the overall garment shown in the image, focusing on its fit, fabric type, sleeve style, neckline, and any notable design elements or features in one or two lines for given image.
                Don't start with "This image shows a pair of beige cargo ..." but instead start with "a pair of beige cargo ..."
            """
        else:
            system_prompt = """
                You are world class fahsion designer
                Your task is to Write a detailed description of the upper body garment shown in the image, focusing on its fit, sleeve style, fabric type, neckline, and any notable design elements or features in one or two lines for given image.
                Don't start with "This image shows a pair of beige cargo ..." but instead start with "a pair of beige cargo ..."
            """

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {
                       "type": "image_url",
                       "image_url": {
                            "url": f"data:image/jpeg;base64,{base_64_image}"
                       }
                    }
                ]},
            ],
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in generate_upper_cloth_description: {e}")
        raise e

def generate_caption_for_image(image):
    """
    Generates a caption for the given image using OpenAI's vision model.
    """
    if image is None:
        return "Please generate a try-on result first."
    
    # Convert the image to base64
    if isinstance(image, str):
        base64_image = pil_image_to_base64(image)
    else:
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    system_prompt = """
        You are a world class campaign generator for cloth that model is wearing.
        Create campaign caption for the image shown below. 
        create engaging campaign captions for products in the merchandise for instagram stories that attract, convert and retain customers.
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating caption: {str(e)}"

def generate_ai_model_prompt(model_description, product_description):
    print("prompt for ai model generation", f" {model_description} wearing {product_description}.")
    return f" {model_description} wearing {product_description}, full image"

def generate_captions(product_description, campaign_context):
    
    #system prompt
    system_prompt = """
        You are a world-class marketing expert.
        Your task is to create engaging, professional, and contextually relevant campaign captions based on the details provided.
        Use creative language to highlight the product's key features and align with the campaign's goals.
        Ensure the captions are tailored to the specific advertising context provided.
    """

    #  user prompt
    user_prompt = f"""
    Campaign Context: {campaign_context}
    Product Description: {product_description}
    Generate captivating captions for this campaign that align with the provided context.
    """
    
    # Call OpenAI API
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    # Extract generated captions
    captions = response.choices[0].message.content.strip()
    return captions


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

              campaign_context = gr.Textbox(
                    label="Describe your campaign context (e.g., 'Summer sale campaign focusing on vibrant colors')",
                    lines=3,
                    placeholder="What message do you want to convey in this campaign?",
                )

              
              with gr.Row():
                    with gr.Column(scale=1, min_width=230):
                        cloth_image = gr.Image(
                            interactive=True, label="Condition Image", type="filepath"
                        )

                        cloth_description = gr.Textbox(
                            label="Cloth Description", 
                            interactive=False, 
                            lines=3
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

                    cloth_image.change(
                            generate_upper_cloth_description,
                            inputs=[cloth_image, cloth_type],
                            outputs=[cloth_description],
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
                # single or multiple image

                result_image = gr.Image(interactive=False, label="Result")
                captions_textbox = gr.Textbox(
                label="Generated Campaign Captions",
                interactive=False,
                lines=6
                )


                with gr.Row():
                    # Photo Examples
                    root_path = "resource/demo/example"
                    

            image_path.change(
                person_example_fn, inputs=image_path, outputs=person_image
            )


            # Connect the generation button
            generate_button.click(
                generate_person_image,
                inputs=[text_prompt, cloth_description],
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
                    campaign_context,
                ],
                [result_image, captions_textbox]
            )
            
            # generate_caption_btn.click(
            #     generate_caption_for_image,
            #     inputs=[result_image],
            #     outputs=[caption_text]
            # )
    demo.queue().launch(share=True, show_error=True)


if __name__ == "__main__":
    app_gradio()
