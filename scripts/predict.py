import numpy as np
from argparse import ArgumentParser
from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import random
import os
import time
import glob
import cv2
from tqdm import tqdm
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt


PERSISTENT_COLOR_MAP: Dict[str, Tuple[float, float, float]] = {}

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook. If your card doesn't support it, try float16 instead
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# inference mode for the whole notebook. Disable if you need gradients
torch.inference_mode().__enter__()


def get_consistent_color(label: str) -> Tuple[float, float, float]:
    """
    Retrieves a consistent color for a given label from the global map,
    generating a new one if the label is new.

    Args:
        label (str): The object label (e.g., "person", "dog").

    Returns:
        Tuple[float, float, float]: The consistent RGB color tuple (0.0-1.0 scale).
    """
    if label in PERSISTENT_COLOR_MAP:
        return PERSISTENT_COLOR_MAP[label]

    # If the label is new, generate a random color and store it
    new_color = (random.random(),
                 random.random(),
                 random.random())

    # Store the new color in the persistent map
    PERSISTENT_COLOR_MAP[label] = new_color

    return new_color


def process_and_combine_detection_results_multi_label(data_list: List[Dict[str, Any]], label_list: List[str]) -> Dict[
    str, Any]:
    """
    Appends a *consistent* 'color' and a unique 'label' to each dictionary in a list and combines them.

    Args:
        data_list (list): A list of dictionaries, where each dict has
                          'boxes' and 'scores' (torch tensors).
        label_list (list): A list of label strings, must have the same length as data_list.

    Returns:
        dict: A single dictionary containing combined 'boxes', 'scores',
              'colors' (0.0-1.0 float tuples), and 'labels'.
    """

    if len(data_list) != len(label_list):
        raise ValueError("The length of 'data_list' must match the length of 'label_list'.")

    all_boxes = []
    all_scores = []
    all_colors = []
    all_labels = []

    for item, label in zip(data_list, label_list):
        # Get the consistent color for this specific label
        consistent_color = get_consistent_color(label)

        # Convert tensors to numpy arrays
        current_boxes = item['boxes'].cpu().float().numpy()
        current_scores = item['scores'].cpu().float().numpy()
        num_boxes = len(current_boxes)

        # Create repeated lists for color and label for all boxes in this item
        # All boxes with the same label in the same dictionary will share this consistent color
        current_colors = [consistent_color] * num_boxes
        current_labels = [label] * num_boxes

        # Append data to the collective lists
        all_boxes.append(current_boxes)
        all_scores.append(current_scores)
        all_colors.extend(current_colors)
        all_labels.extend(current_labels)

    if not all_boxes:
        return {'boxes': np.array([]), 'scores': np.array([]), 'colors': [], 'labels': []}

    combined_boxes = np.concatenate(all_boxes, axis=0)
    combined_scores = np.concatenate(all_scores, axis=0)

    # Final single dictionary
    combined_data = {
        'boxes': combined_boxes,
        'scores': combined_scores,
        'colors': all_colors,
        'labels': all_labels
    }

    return combined_data


def plot_boxes_with_opencv(image_path, combined_data: Dict[str, Any], output_folder: str,
                        output_filename: str = "output_detections.jpg"):
    # 1. Load the Image and Convert to OpenCV format (BGR, default for cv2)
    # image_np = np.array(image)
    if isinstance(image_path, str):
        image_bgr = cv2.imread(image_path)
    else:
        image_bgr = image_path[:, :, ::-1].copy()

    boxes = combined_data['boxes']
    scores = combined_data['scores']
    colors = combined_data['colors']
    labels = combined_data['labels']

    for box, score, color_norm, label in zip(boxes, scores, colors, labels):
        # Denormalize color from 0.0-1.0 to 0-255 and convert to BGR tuple
        # Matplotlib uses RGB (1.0, 0.0, 0.0) -> OpenCV uses BGR (0, 0, 255)
        color_bgr_255 = (int(color_norm[2] * 255),  # B
                         int(color_norm[1] * 255),  # G
                         int(color_norm[0] * 255))  # R

        x1, y1, x2, y2 = map(int, box)  # Boxes must be integers for cv2 drawing

        # Draw Bounding Box (Rectangle)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color_bgr_255, 2)

        # Draw Text Label
        display_label = f"{label} {score:.2f}"

        # Optionally, draw a filled rectangle for the text background (more complex in cv2)
        # For simplicity here, just draw text directly:
        cv2.putText(image_bgr, display_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr_255, 2)

    # 4. Save the image
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, image_bgr)
    # print(f"✅ Annotated image saved successfully to: {output_path}")

def plot_boxes_on_image(image, combined_data: Dict[str, Any], output_folder: str,
                        output_filename: str = "output_detections.jpg"):
    """
    Plots bounding boxes, scores, and labels onto a PIL image and saves the result.

    Args:
        image_path (str): Path to the input image file.
        combined_data (dict): Dictionary containing 'boxes', 'scores', 'colors', and 'labels'.
                              'boxes' should be a NumPy array of shape (N, 4) in [x1, y1, x2, y2] format.
                              'colors' should be a list of (R, G, B) tuples (0.0-1.0 scale).
        output_folder (str): Directory where the annotated image will be saved.
        output_filename (str): Name for the saved annotated image file.
    """

    # 2. Setup Matplotlib figure and axes
    # Convert image to numpy array for Matplotlib to display
    image_np = np.array(image)

    # Create figure and axes with the size matching the image aspect ratio
    fig, ax = plt.subplots(1)

    # Display the image as the background
    ax.imshow(image_np)

    # 3. Extract data
    boxes = combined_data['boxes']
    scores = combined_data['scores']
    colors = combined_data['colors']
    labels = combined_data['labels']

    # Quick check for data consistency
    if not (len(boxes) == len(scores) == len(colors) == len(labels)):
        print("Error: Data arrays/lists in combined_data have inconsistent lengths.")
        return

    # 4. Loop through each detection and plot it
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        color = colors[i]
        label = labels[i]

        # Bounding box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = box

        # Calculate width and height for Matplotlib Rectangle patch
        width = x2 - x1
        height = y2 - y1

        # Create a Rectangle patch for the bounding box
        rect = plt.Rectangle((x1, y1), width, height,
                             linewidth=2,
                             edgecolor=color,
                             facecolor='none',
                             alpha=0.8)  # Semi-transparent edge

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Format the label text (e.g., "Person 0.95")
        display_label = f"{label} {score:.2f}"

        # Add text label for the detection
        ax.text(x1, y1 - 5,  # Position slightly above the top-left corner
                display_label,
                color='white',
                fontsize=8,
                # Create a background box for the text
                bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2))

    # 5. Finalize plot settings
    ax.axis('off')  # Hide the axis ticks and labels for a clean image
    # Ensure the plot area fits the image exactly
    ax.set_xlim(0, image_np.shape[1])
    ax.set_ylim(image_np.shape[0], 0)  # Matplotlib y-axis is inverted relative to image coordinates
    fig.tight_layout(pad=0)

    # 6. Save the image
    output_path = os.path.join(output_folder, output_filename)

    # Save the figure, making sure only the image area is saved
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free up memory

    # print(f"\n✅ Annotated image saved successfully to: {output_path}")

def main():
    # Image 1, we'll use two text prompts
    parser = ArgumentParser()
    parser.add_argument('video_path', help='path to the video to be analyzed', default="../assets/videos/0001")
    parser.add_argument('prompt', help='a text prompt describing the objects to look for', nargs='+', default=["person"])

    args = parser.parse_args()
    video_path = args.video_path

    # create output folder
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        output_dir = video_path[:-4]

    else:
        output_dir = video_path + '_res'

    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

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
    print('Done reading the video in {:.2f} s.'.format(end))
    print('*' * 33)

    print('Start analyzing video:')
    print('*' * len('Start analyzing video:'))
    start = time.time()

    prompts = args.prompt

    frame_idx = 0
    for frame in tqdm(video_frames_for_vis):

        if isinstance(frame, str):
            img = Image.open(frame).convert('RGB')
        else:
            img = Image.fromarray(frame, 'RGB')

        # Pre-process image and compute vision embeddings once
        img_inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

        all_results = []
        for prompt in prompts:
            text_inputs = processor(text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(vision_embeds=vision_embeds, **text_inputs)

            results = processor.post_process_object_detection(
                outputs,
                threshold=0.5,
                target_sizes=img_inputs.get("original_sizes").tolist()
            )[0]
            all_results.append(results)

        if len(all_results):
            all_results = process_and_combine_detection_results_multi_label(all_results, prompts)
            filename = os.path.basename(frame) if isinstance(frame, str) else f"frame_{frame_idx}.png"
            plot_boxes_with_opencv(frame, all_results, output_dir, filename)

        frame_idx += 1

    end = time.time() - start
    print('Analysis of {} frames took {:.2f} s.'.format(len(video_frames_for_vis),end))

if __name__ == '__main__':
    main()