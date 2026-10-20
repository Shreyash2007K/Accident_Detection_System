"""
Standalone utility to generate realistic sample traffic test videos.
Generates:
1. sample_videos/traffic_collision.mp4 (Simulated lane intrusion and accident event)
2. sample_videos/traffic_normal.mp4 (Smooth regular traffic flow without collisions)
"""

import os
from pathlib import Path
from utils import generate_sample_traffic_video

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_videos"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("[INFO] Generating sample traffic videos for testing and demo...")

    collision_path = str(SAMPLE_DIR / "traffic_collision.mp4")
    print(f"Creating collision demo video: {collision_path}")
    generate_sample_traffic_video(collision_path, scenario="collision", duration_sec=6, fps=24)
    print(" -> Collision scenario generated successfully.")

    normal_path = str(SAMPLE_DIR / "traffic_normal.mp4")
    print(f"Creating normal traffic demo video: {normal_path}")
    generate_sample_traffic_video(normal_path, scenario="normal", duration_sec=6, fps=24)
    print(" -> Normal traffic scenario generated successfully.")

    print("\n[SUCCESS] Sample videos created successfully in:", SAMPLE_DIR)

if __name__ == "__main__":
    main()
