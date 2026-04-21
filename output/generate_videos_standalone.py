#!/usr/bin/env python3
"""
Standalone video generator for Crash-1500 descriptions.
Uses FramePack inference pipeline directly without Gradio.
"""

import os
import sys
import json
import time
import uuid
import argparse
from pathlib import Path

# Set environment
os.environ['HF_HOME'] = os.path.abspath(os.path.join(os.path.dirname(__file__), './hf_download'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
from PIL import Image

# Add FramePack to path
sys.path.insert(0, '/root/autodl-tmp/FramePack-Studio-main')

from diffusers_helper.hf_login import login
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode
from diffusers_helper.utils import save_bcthw_as_mp4, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from diffusers_helper.bucket_tools import find_nearest_bucket

def load_models():
    """Load all required models"""
    print("Loading models...")
    
    # Text encoders
    text_encoder = LlamaModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", 
        subfolder='text_encoder',
        torch_dtype=torch.float16
    ).cpu()
    
    text_encoder_2 = CLIPTextModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='text_encoder_2', 
        torch_dtype=torch.float16
    ).cpu()
    
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='tokenizer'
    )
    
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='tokenizer_2'
    )
    
    # VAE
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder='vae',
        torch_dtype=torch.float16
    ).cpu()
    
    # Transformer
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        "lllyasviel/FramePackI2V_HY",
        torch_dtype=torch.bfloat16
    ).cpu()
    
    print("Models loaded!")
    return text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, transformer

def generate_video(
    prompt,
    output_path,
    seed=2500,
    steps=25,
    cfg=1,
    gs=10,
    total_seconds=3,
    resolution=640,
    device='cuda'
):
    """Generate a video from a prompt"""
    
    # Load models
    text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, transformer = load_models()
    
    # Move to GPU
    text_encoder = text_encoder.to(device)
    text_encoder_2 = text_encoder_2.to(device)
    vae = vae.to(device)
    transformer = transformer.to(device)
    
    # Encode prompt
    print(f"Encoding prompt: {prompt[:60]}...")
    llama_vec, clip_l_pooler = encode_prompt_conds(
        prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
    )
    
    # Setup generation parameters
    latent_window_size = 9
    fps = 30
    total_frames = int(total_seconds * fps)
    
    # Calculate latent dimensions
    height, width = resolution, resolution
    latent_h = height // 8
    latent_w = width // 8
    latent_frames = (total_frames - 1) // 4 + 1
    
    # Generate noise
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, transformer.config.in_channels, latent_frames, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=torch.bfloat16
    )
    
    # Sample
    print("Generating video...")
    from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
    
    def model_fn(x, timestep, **kwargs):
        return transformer(
            x,
            timestep=timestep,
            encoder_hidden_states=llama_vec,
            encoder_hidden_states_2=clip_l_pooler,
            **kwargs
        ).sample
    
    # Simple sampling (using euler)
    from k_sampling import get_sigmas
    sigmas = get_sigmas(steps, device=device)
    
    for i, sigma in enumerate(sigmas[:-1]):
        sigma_in = sigma
        sigma_out = sigmas[i + 1]
        
        # Denoise
        with torch.no_grad():
            noise_pred = model_fn(latents, sigma_in)
        
        # Euler step
        latents = latents + (sigma_out - sigma_in) * noise_pred
        
        if i % 5 == 0:
            print(f"  Step {i+1}/{steps}")
    
    # Decode
    print("Decoding video...")
    video = vae_decode(latents, vae)
    
    # Save
    print(f"Saving to {output_path}...")
    save_bcthw_as_mp4(video, output_path, fps=fps)
    
    # Cleanup
    del text_encoder, text_encoder_2, vae, transformer
    torch.cuda.empty_cache()
    
    print(f"Done! Saved to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate videos from descriptions')
    parser.add_argument('--descriptions', type=str, required=True, help='Path to classified_descriptions.json')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/generated_videos', help='Output directory')
    parser.add_argument('--max_videos', type=int, default=10, help='Max videos to generate')
    parser.add_argument('--categories', type=str, default='safe,near_crash,crash', help='Categories to generate')
    
    args = parser.parse_args()
    
    # Load descriptions
    with open(args.descriptions, 'r') as f:
        descriptions = json.load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse categories
    categories = args.categories.split(',')
    
    # Generate videos
    count = 0
    for video_id in sorted(descriptions.keys())[:args.max_videos]:
        for category in categories:
            if count >= args.max_videos * len(categories):
                break
                
            prompt = descriptions[video_id][category]
            output_path = os.path.join(args.output_dir, f"{video_id}_{category}.mp4")
            
            if os.path.exists(output_path):
                print(f"Skipping {output_path} (already exists)")
                continue
            
            print(f"\n{'='*60}")
            print(f"Generating {video_id}/{category}")
            print(f"Prompt: {prompt[:80]}...")
            print(f"{'='*60}")
            
            try:
                generate_video(
                    prompt=prompt,
                    output_path=output_path,
                    seed=2500 + count,
                    steps=25,
                    total_seconds=3,
                    resolution=640
                )
                count += 1
            except Exception as e:
                print(f"Error generating {video_id}/{category}: {e}")
                continue
    
    print(f"\nGenerated {count} videos in {args.output_dir}")

if __name__ == '__main__':
    main()
