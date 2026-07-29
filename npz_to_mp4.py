import numpy as np
import cv2
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_file", help="Path to npz file")
    parser.add_argument("out_mp4", help="Path to output mp4 file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    data = np.load(args.npz_file)
    obs = data['obs'] # shape (N, H, W, C)
    
    N, H, W, C = obs.shape
    
    # We will resize to make it more visible, 64x64 is tiny
    out_H, out_W = H * 8, W * 8
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out_mp4, fourcc, args.fps, (out_W, out_H))
    
    for i in range(N):
        frame = obs[i]
        # In case the frame is RGB, we should convert to BGR for cv2
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Resize
        frame = cv2.resize(frame, (out_W, out_H), interpolation=cv2.INTER_NEAREST)
        out.write(frame)
        
    out.release()
    print(f"Video saved to {args.out_mp4}")

if __name__ == "__main__":
    main()
