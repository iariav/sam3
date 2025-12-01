import os

import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

import torch
import time

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path)

# for loading a saved checkpoint
# model = build_sam3_image_model(enable_inst_interactivity=True, load_from_HF=False) # only build the model
# checkpoint = torch.load(checkpoint_path, map_location="cpu")
# model = sam3.train.utils.checkpoint_utils.load_state_dict_into_model(model=model, state_dict=checkpoint["model"])

def main():

    image_path = "/hdd_data1/VisDrone/VisDrone2019-DET-test-dev/images/9999973_00000_d_0000055.jpg"
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # config = Sam3Config.from_pretrained("facebook/sam3")
    # config.vision_config.backbone_config.image_size = INFERENCE_RESOLUTION # 1400 or any multiple of patch_size (14)
    # model = Sam3Model.from_pretrained("facebook/sam3", config=config).to(device)
    # processor = Sam3Processor.from_pretrained("facebook/sam3")

    processor = Sam3Processor(model, confidence_threshold=0.5)
    inference_state = processor.set_image(image)

    prompts = ["solar panel","window","car","road","bush"]
    for prompt in prompts:
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)
        start = time.time()
        img0 = Image.open(image_path)
        plot_results(img0, inference_state)
        end = time.time() - start
        print('Analysis took {} s'.format(end))
        plt.show()


if __name__ == '__main__':
    main()