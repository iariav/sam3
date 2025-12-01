from PIL import Image
import requests
from io import BytesIO
from transformers import Sam3Processor, Sam3Model
import numpy as np
import matplotlib
import torch
from PIL import Image
import os
from typing import List
import time
import glob
import cv2
from tqdm import tqdm
import gc
# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook. If your card doesn't support it, try float16 instead
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# inference mode for the whole notebook. Disable if you need gradients
torch.inference_mode().__enter__()


from sam3.visualization_utils import plot_results

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")


def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)

    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

def main():
    # Image 1, we'll use two text prompts

    video_path = "/ssd_data/edge/Demo/2.mp4"

    print("Reading video file {}:".format(video_path))
    print('*' * (len(video_path) + 20))
    start = time.time()
    # load "video_frames_for_vis" for visualization purposes (they are not used by the model)
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        process_every_n_frames = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        video_frames_for_vis = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % process_every_n_frames == 0:
                video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame_count += 1

        cap.release()
    else:
        video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
        try:
            # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
            video_frames_for_vis.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
        except ValueError:
            # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
            print(
                f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
                f"falling back to lexicographic sort."
            )
            video_frames_for_vis.sort()

    end = time.time() - start
    print('Done reading the video in {:2} s.'.format(end))
    print('*' * 40)
    num_batched_images = 4

    print('Start analyzing video:')
    print('*' * len('Start analyzing video:'))
    start = time.time()

    prompts = ["a red car", "person"]

    for frame in tqdm(video_frames_for_vis):

        if isinstance(frame, str):
            img = Image.open(frame)
        else:
            img = Image.fromarray(frame, 'RGB')

        images = [img] * len(prompts)  # This doesn't duplicate the actual image data
        # Collate then move to cuda
        inputs = processor(images=images, text=prompts, return_tensors="pt").to(device)

        # Forward. Note that the first forward will be very slow due to compilation
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process results for both images
        results = processor.post_process_object_detection(
            outputs,
            threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )

        if len(results):
            plot_results(img, results[0])

        # del batch
        # gc.collect()
        # torch.cuda.empty_cache()

        #
        # plot_results(img1, processed_results[id2])
        #
        # plot_results(img2, processed_results[id3])
        #
        # plot_results(img2, processed_results[id4])

    end = time.time() - start
    print('Analysis of {} frames took {} s with batch size {}.'.format(len(video_frames_for_vis),end,num_batched_images))
if __name__ == '__main__':
    main()

    # img_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    # image = Image.open(requests.get(img_url, stream=True).raw)
    # img_inputs = processor(images=image, return_tensors="pt")
    #
    # # Pre-compute vision embeddings
    # vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)
    #
    # text_prompts = ["a person", "a car", "a dog"]
    # all_masks = []
    # all_scores = []
    #
    # for prompt in text_prompts:
    #     text_inputs = processor(text=prompt, return_tensors="pt")
    #     with torch.no_grad():
    #         outputs = model(vision_embeds=vision_embeds, **text_inputs)
    #
    #     masks = processor.post_process_masks(
    #         outputs.pred_masks,
    #         inputs["original_sizes"],
    #     )
    #     scores = outputs.iou_scores
    #
    #     all_masks.append(masks)
    #     all_scores.append(scores)