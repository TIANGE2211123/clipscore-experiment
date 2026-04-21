#!/usr/bin/env python3
"""
Generate video queue jobs from classified descriptions.
Creates jobs for FramePack-Studio to generate safe/near-crash/crash videos.
"""

import json
import uuid
import time
import os

# Load descriptions
with open('/root/autodl-tmp/FramePack-Studio-main/classified_descriptions.json', 'r') as f:
    descriptions = json.load(f)

# Load existing queue
queue_path = '/root/autodl-tmp/FramePack-Studio-main/queue.json'
with open(queue_path, 'r') as f:
    queue = json.load(f)

# Default parameters for video generation
default_params = {
    "app_version": "0.5.1",
    "negative_prompt": "",
    "seed": 2500,
    "steps": 25,
    "cfg": 1,
    "gs": 10,
    "rs": 0,
    "latent_type": "Noise",
    "resolutionW": 640,
    "resolutionH": 640,
    "model_type": "Original with Endframe",
    "generation_type": "Original with Endframe",
    "has_input_image": False,
    "input_image_path": None,
    "total_second_length": 3,
    "blend_sections": 4,
    "latent_window_size": 9,
    "num_cleaned_frames": 5,
    "end_frame_strength": 0.05,
    "end_frame_image_path": None,
    "end_frame_used": "True",
    "input_video": None,
    "video_path": None,
    "x_param": None,
    "y_param": None,
    "x_values": None,
    "y_values": None,
    "combine_with_source": True,
    "use_teacache": False,
    "teacache_num_steps": 25,
    "teacache_rel_l1_thresh": 0.15,
    "use_magcache": True,
    "magcache_threshold": 0.1,
    "magcache_max_consecutive_skips": 2,
    "magcache_retention_ratio": 0.25,
    "loras": {},
    "status": "pending",
    "created_at": time.time(),
    "started_at": None,
    "completed_at": None,
    "error": None,
    "result": None,
    "queue_position": None,
    "saved_input_image_path": None,
    "saved_end_frame_image_path": None
}

# Generate jobs for first 10 videos (to test)
# Change this to process all videos
test_videos = list(descriptions.keys())[:10]

new_jobs = 0
for video_id in test_videos:
    desc = descriptions[video_id]
    
    # Create 3 jobs: safe, near_crash, crash
    for category in ['safe', 'near_crash', 'crash']:
        job_id = str(uuid.uuid4())
        
        # Create job with parameters
        job = default_params.copy()
        job['id'] = job_id
        job['prompt'] = desc[category]
        job['timestamp'] = time.time()
        
        # Add to queue
        queue[job_id] = job
        new_jobs += 1
        
        print(f"Created job {job_id} for {video_id}/{category}")

# Save updated queue
with open(queue_path, 'w') as f:
    json.dump(queue, f, indent=2)

print(f"\nAdded {new_jobs} jobs to queue")
print(f"Total jobs in queue: {len(queue)}")
