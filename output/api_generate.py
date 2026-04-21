#!/usr/bin/env python3
"""
Add video generation jobs via Gradio API.
"""

import json
import time
from gradio_client import Client

# Load descriptions
with open('/root/autodl-tmp/FramePack-Studio-main/classified_descriptions.json', 'r') as f:
    descriptions = json.load(f)

# Connect to FramePack-Studio
print("Connecting to FramePack-Studio...")
client = Client('http://localhost:7860')
print("Connected!")

# Get first 5 videos
video_ids = sorted(descriptions.keys())[:5]
categories = ['safe', 'near_crash', 'crash']

print(f"\nWill generate {len(video_ids) * len(categories)} videos")
print(f"Videos: {video_ids}")

# Try to find the process endpoint
api = client.view_api()
process_endpoint = None

for name in api['named_endpoints'].keys():
    if name == '/process':
        process_endpoint = name
        break

if not process_endpoint:
    # Try alternative names
    for name in api['named_endpoints'].keys():
        if 'process' in name.lower():
            process_endpoint = name
            print(f"Found alternative endpoint: {name}")
            break

if not process_endpoint:
    print("ERROR: Could not find process endpoint!")
    print("Available endpoints:")
    for name in sorted(api['named_endpoints'].keys())[:20]:
        print(f"  {name}")
    exit(1)

print(f"Using endpoint: {process_endpoint}")

# Generate videos
for video_id in video_ids:
    for category in categories:
        prompt = descriptions[video_id][category]
        
        print(f"\n{'='*60}")
        print(f"Generating {video_id}/{category}")
        print(f"Prompt: {prompt[:80]}...")
        
        try:
            # Call the process endpoint
            result = client.predict(
                prompt,  # prompt
                "",  # negative_prompt
                2500,  # seed
                25,  # steps
                1.0,  # cfg
                10.0,  # gs
                0.0,  # rs
                "Noise",  # latent_type
                640,  # resolutionW
                640,  # resolutionH
                "Original with Endframe",  # model_type
                3,  # total_second_length
                4,  # blend_sections
                9,  # latent_window_size
                5,  # num_cleaned_frames
                0.05,  # end_frame_strength
                True,  # end_frame_used
                None,  # input_image
                None,  # end_frame_image
                False,  # use_teacache
                25,  # teacache_num_steps
                0.15,  # teacache_rel_l1_thresh
                True,  # use_magcache
                0.1,  # magcache_threshold
                2,  # magcache_max_consecutive_skips
                0.25,  # magcache_retention_ratio
                {},  # loras
                api_name=process_endpoint
            )
            print(f"Result: {result}")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        time.sleep(2)

print("\nDone!")
