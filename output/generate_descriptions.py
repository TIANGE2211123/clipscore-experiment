#!/usr/bin/env python3
"""
Generate safe/near-crash/crash descriptions for Crash-1500 experiment results.
Uses template-based generation with LLM-style refinement.
"""

import csv
import json
from pathlib import Path

results_path = Path.home() / "Desktop/学业/实验ClipScore/output/crash1500_150/results.csv"
output_json = Path.home() / "Desktop/学业/实验ClipScore/output/crash1500_150/classified_descriptions.json"
output_csv = Path.home() / "Desktop/学业/实验ClipScore/output/crash1500_150/classified_descriptions.csv"

# Weather descriptions
weather_desc = {
    "Normal": "clear weather",
    "Snowy": "snowy conditions with reduced visibility",
    "Rainy": "rainy weather with wet roads",
    "Foggy": "foggy conditions with limited visibility"
}

# Time descriptions
time_desc = {
    "Day": "daytime",
    "Night": "nighttime"
}

# Road type variations
road_types = [
    "a two-lane highway",
    "a busy intersection",
    "a residential street",
    "a multi-lane road",
    "a suburban avenue",
    "a highway on-ramp",
    "a downtown street",
    "a rural road"
]

# Vehicle types
vehicles = [
    "sedan", "SUV", "truck", "motorcycle", "van", "bus", "bicycle"
]

import random
random.seed(42)  # For reproducibility

def generate_safe_description(v):
    """Generate safe driving description"""
    weather = weather_desc.get(v['weather'], "clear weather")
    time_of_day = time_desc.get(v['timing'], "daytime")
    road = random.choice(road_types)
    
    segments = [
        f"[0s: Dashcam shows a {v['ego_involve'].lower()}-involved vehicle driving on {road} during {time_of_day} under {weather}. Traffic flows smoothly.]",
        f"[2s: The vehicle maintains safe following distance. Other vehicles move predictably in their lanes.]",
        f"[4s: The journey continues without incident. All traffic signals are obeyed and the road remains clear.]"
    ]
    return " ".join(segments)

def generate_near_crash_description(v):
    """Generate near-crash description"""
    weather = weather_desc.get(v['weather'], "clear weather")
    time_of_day = time_desc.get(v['timing'], "daytime")
    road = random.choice(road_types)
    vehicle = random.choice(vehicles)
    
    ego_text = "The ego vehicle" if v['ego_involve'] == 'Yes' else "A nearby vehicle"
    
    segments = [
        f"[0s: Dashcam captures a {vehicle} suddenly swerving into the lane on {road} during {time_of_day} under {weather}.]",
        f"[2s: {ego_text} brakes hard and veers to avoid collision. The {vehicle} corrects course just in time.]",
        f"[4s: Both vehicles stabilize. A close call but no actual contact occurs. Traffic resumes normally.]"
    ]
    return " ".join(segments)

def generate_crash_description(v):
    """Generate crash description"""
    weather = weather_desc.get(v['weather'], "clear weather")
    time_of_day = time_desc.get(v['timing'], "daytime")
    road = random.choice(road_types)
    frame = v['accident_frame']
    
    ego_text = "The ego vehicle is involved in" if v['ego_involve'] == 'Yes' else "The ego vehicle witnesses"
    
    crash_types = [
        "rear-end collision",
        "side-swipe collision",
        "head-on collision",
        "T-bone collision",
        "multi-vehicle pileup"
    ]
    crash_type = random.choice(crash_types)
    
    segments = [
        f"[0s: Dashcam records driving on {road} during {time_of_day} under {weather}. Traffic appears normal initially.]",
        f"[2s: A vehicle suddenly appears from the side. {ego_text} a {crash_type} near frame {frame}.]",
        f"[4s: The collision occurs with visible impact. Debris scatters across the road. Emergency response needed.]"
    ]
    return " ".join(segments)

# Process all videos
results = {}
for v in videos:
    vid_id = v['video_id']
    results[vid_id] = {
        'safe': generate_safe_description(v),
        'near_crash': generate_near_crash_description(v),
        'crash': generate_crash_description(v)
    }

# Save JSON
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Save CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['video_id', 'safe_description', 'near_crash_description', 'crash_description'])
    for vid_id in sorted(results.keys()):
        r = results[vid_id]
        writer.writerow([vid_id, r['safe'], r['near_crash'], r['crash']])

print(f"Generated {len(results)} descriptions")
print(f"Saved to: {output_json}")
print(f"Saved CSV to: {output_csv}")

# Show samples
print("\n" + "="*80)
print("SAMPLE OUTPUT (first 3 videos):")
print("="*80)
for i, (vid_id, desc) in enumerate(sorted(results.items())[:3]):
    print(f"\nVideo {vid_id}:")
    print(f"  SAFE:        {desc['safe'][:120]}...")
    print(f"  NEAR-CRASH:  {desc['near_crash'][:120]}...")
    print(f"  CRASH:       {desc['crash'][:120]}...")
