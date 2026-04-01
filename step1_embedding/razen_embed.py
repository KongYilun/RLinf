import cv2
from PIL import Image
import numpy as np
import sys
from rzen_embed_inference import RzenEmbed

def extract_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            break
    cap.release()
    return frames

rzen = RzenEmbed("/data/users/kongyilun/models/RzenEmbed")

queries = [
    "A traditional boat glides along a river lined with blooming cherry blossoms under an overcast sky in a modern cityscape.",
    "Tiny ginger kitten meows cutely by the water."
]

# Extract frames from videos
video_path_list = [
    "assets/example5.mp4",
    "assets/example6.mp4",
]
candidates = [extract_frames(video_path, num_frames=8) for video_path in video_path_list]

query_instruction = "Find the video snippet that corresponds to the given caption: "
candidate_instruction = "Understand the content of the provided video."

# Generate embeddings for video retrieval
query_embeds = rzen.get_fused_embeddings(instruction=query_instruction, texts=queries)
candidate_embeds = rzen.get_fused_embeddings(instruction=candidate_instruction, images=candidates)

# Calculate text-to-video similarity scores
similarity_scores = query_embeds @ candidate_embeds.T
print(similarity_scores)