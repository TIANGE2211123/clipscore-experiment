#!/usr/bin/env python3
"""
Direct queue manipulation - add jobs and trigger processing.
"""

import json
import sys
import os

# Add FramePack to path
sys.path.insert(0, '/root/autodl-tmp/FramePack-Studio-main')
os.chdir('/root/autodl-tmp/FramePack-Studio-main')

# Import the queue module
from modules.video_queue import VideoJobQueue, JobStatus
from modules.pipelines.worker import worker

# Load descriptions
with open('classified_descriptions.json', 'r') as f:
    descriptions = json.load(f)

# Load existing queue
queue_path = 'queue.json'
with open(queue_path, 'r') as f:
    existing_queue = json.load(f)

print(f"Existing queue: {len(existing_queue)} jobs")

# Create new jobs
new_jobs = {}
video_ids = sorted(descriptions.keys())[:10]
categories = ['safe', 'near_crash', 'crash']

import uuid
import time

for video_id in video_ids:
    for category in categories:
        job_id = str(uuid.uuid4())
        prompt = descriptions[video_id][category]
        
        job = {
            "app_version": "0.5.1",
            "prompt": prompt,
            "negative_prompt": "",
            "seed": 2500 + hash(job_id) % 10000,
            "steps": 25,
            "cfg": 1,
            "gs": 10,
            "rs": 0,
            "latent_type": "Noise",
            "timestamp": time.time(),
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
            "id": job_id,
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
        
        new_jobs[job_id] = job
        print(f"Created: {video_id}/{category} -> {job_id}")

# Save updated queue
existing_queue.update(new_jobs)
with open(queue_path, 'w') as f:
    json.dump(existing_queue, f, indent=2)

print(f"\nAdded {len(new_jobs)} new jobs")
print(f"Total queue: {len(existing_queue)} jobs")
print(f"Pending: {sum(1 for j in existing_queue.values() if j['status'] == 'pending')}")
