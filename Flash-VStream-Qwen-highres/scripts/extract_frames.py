import decord
import torch
import os
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

min_frames = 4
max_frames = 14400
fps = 1

INPUT_DIR = '/scratch/zwu24/datasets/LVBench/scripts/videos'
OUTPUT_DIR = '/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/data/eval_video/lvbench/frames'

# INPUT_DIR = '/scratch/zwu24/datasets/MLVU/MLVU/video'
# OUTPUT_DIR = '/scratch/zwu24/Flash-VStream-highres/Flash-VStream-Qwen-highres/data/eval_video/MLVU/frames'

VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def process_video(file_path):
    video_id = os.path.splitext(os.path.basename(file_path))[0]
    video_path = os.path.join(OUTPUT_DIR, video_id)

    try:
        vr = decord.VideoReader(file_path)
        video_len = len(vr)

        if video_len <= 0:
            return

        duration = video_len / max(vr.get_avg_fps(), 1e-6)
        nframes = round(duration * fps)
        nframes = min(max(nframes, min_frames), max_frames)

        safe_end = max(0, video_len - 2)
        idx = torch.linspace(0, safe_end, nframes).round().long().clamp(0, safe_end)

        chunk = 256
        frames = []

        for i in range(0, len(idx), chunk):
            part = idx[i:i + chunk].tolist()
            frames.append(vr.get_batch(part).asnumpy())

        frames = torch.cat([torch.from_numpy(x) for x in frames], dim=0).numpy()

        os.makedirs(video_path, exist_ok=True)

        for i in range(len(frames)):
            Image.fromarray(frames[i]).save(
                os.path.join(video_path, f'{i:06d}.jpg'),
                quality=100
            )

    except Exception as e:
        print(f"\nBAD VIDEO: {file_path}")
        print(e)
        with open("bad_videos.txt", "a") as f:
            f.write(file_path + "\n")


def collect_files_unique(root):
    files = []
    seen = set()

    for r, _, fs in os.walk(root):
        for f in fs:
            ext = os.path.splitext(f)[1].lower()
            if ext not in VIDEO_EXT:
                continue

            video_id = os.path.splitext(f)[0]

            if video_id in seen:
                continue

            seen.add(video_id)
            files.append(os.path.join(r, f))

    return files


if __name__ == '__main__':
    files = collect_files_unique(INPUT_DIR)

    os.environ["DECORD_EOF_RETRY_MAX"] = "50000"

    with Pool(processes=16) as pool:
        list(tqdm(pool.imap_unordered(process_video, files), total=len(files)))