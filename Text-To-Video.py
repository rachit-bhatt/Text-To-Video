import torch
from diffusers import TextToVideoZeroPipeline
import imageio
import numpy as np

FPS = 4
SECONDS = 6

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
seed = 0
#video_length = 24  #24 ÷ 4fps = 6 seconds
video_length = FPS * SECONDS
chunk_size = 8
# prompt = "Emilia Clarke is tied up using various machines that tickle her. Her hands and legs are stretched wide. The machine includes various hands, fur, and vibrators to tickle her. She is wearing almost no clothes and is laughing hard while she is being tickled. A few vibrators are also tickling on her sensitive body parts."
prompt = 'Emilia Clarke is being tickled by her four friends with her concent and she is enjoying the laugh.'

# Generate the video chunk-by-chunk
result = []
chunk_ids = np.arange(0, video_length, chunk_size - 1)
generator = torch.Generator(device="cuda")
for i in range(len(chunk_ids)):
    print(f"Processing chunk {i + 1} / {len(chunk_ids)}")
    ch_start = chunk_ids[i]
    ch_end = video_length if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
    # Attach the first frame for Cross Frame Attention
    frame_ids = [0] + list(range(ch_start, ch_end))
    # Fix the seed for the temporal consistency
    generator.manual_seed(seed)
    output = pipe(prompt=prompt, video_length=len(frame_ids), generator=generator, frame_ids=frame_ids)
    result.append(output.images[1:])

# Concatenate chunks and save
result = np.concatenate(result)
result = [(r * 255).astype("uint8") for r in result]
imageio.mimsave("Output.mp4", result)