import json
import os
import sys
import tqdm
from omegaconf import OmegaConf

import lpips as lpips_lib

import torch
from torch.utils.data import DataLoader

from gaussian_renderer import render_predicted
from scene.dataset_factory import get_dataset
from utils.loss_utils import ssim as ssim_fn
from scene.hmr2_extension import load_hmr_predictor


class Metricator():
    def __init__(self, device):
        self.lpips_net = lpips_lib.LPIPS(net='vgg').to(device)
    
    def compute_metrics(self, image, target):
        lpips = self.lpips_net( image.unsqueeze(0) * 2 - 1, target.unsqueeze(0) * 2 - 1).item()
        psnr = -10 * torch.log10(torch.mean((image - target) ** 2, dim=[0, 1, 2])).item()
        ssim = ssim_fn(image, target).item()
        return psnr, ssim, lpips


@torch.no_grad()
def evaluate_dataset(model, dataloader, device, model_cfg):
    """
    Runs evaluation on the dataset passed in the dataloader. 
    Computes, prints and saves PSNR, SSIM, LPIPS.
    """

    with open("scores.txt", "w+") as f:
        f.write("")

    bg_color = [1, 1, 1] if model_cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # instantiate metricator
    metricator = Metricator(device)

    psnr_all_examples_novel = []
    ssim_all_examples_novel = []
    lpips_all_examples_novel = []
    psnr_all_examples_sherf_novel = []

    psnr_all_examples_cond = []
    ssim_all_examples_cond = []
    lpips_all_examples_cond = []
    psnr_all_examples_sherf_cond = []

    for d_idx, data in enumerate(tqdm.tqdm(dataloader)):
        psnr_all_renders_novel = []
        ssim_all_renders_novel = []
        lpips_all_renders_novel = []
        psnr_all_renders_sherf_novel = []
        psnr_all_renders_cond = []
        ssim_all_renders_cond = []
        lpips_all_renders_cond = []
        psnr_all_renders_sherf_cond = []

        data = {k: v.to(device) for k, v in data.items()}

        rot_transform_quats = data["source_cv2wT_quat"][:, :model_cfg.data.input_images]

        pps_pixels_pred = data["pps_pixels"][:, :model_cfg.data.input_images, ...]

        focals_pixels_pred = data["focals_pixels"][:, :model_cfg.data.input_images, ...]

        input_images = data["gt_images"][:, :model_cfg.data.input_images, ...]

        example_id = dataloader.dataset.get_example_id(d_idx)

        # batch has length 1, the first image is conditioning
        reconstruction, _ = model(
                input_images,
                data["view_to_world_transforms"][:, :model_cfg.data.input_images, ...],
                                rot_transform_quats,
                                focals_pixels_pred,
                                pps_pixels_pred)

        gt_images = []
        rendered = []
        for r_idx in range(1, data["gt_images"].shape[1]):
            if "background_color" in data.keys():
                background_color = data["background_color"][0, r_idx]
            else:
                background_color = background

            if "focals_pixels" in data.keys():
                focals_pixels_render = data["focals_pixels"][0, r_idx]
            else:
                focals_pixels_render = None
            image = render_predicted({k: v[0].contiguous() for k, v in reconstruction.items()},
                                     data["world_view_transforms"][0, r_idx],
                                     data["full_proj_transforms"][0, r_idx], 
                                     data["camera_centers"][0, r_idx],
                                     background_color,
                                     model_cfg,
                                     focals_pixels=focals_pixels_render)["render"]

            gt_images.append(data["gt_images"][0, r_idx, ...])
            rendered.append(image)

            # exclude non-foreground images from metric computation
            if not torch.all(data["gt_images"][0, r_idx, ...] == 0):
                psnr, ssim, lpips = metricator.compute_metrics(image, data["gt_images"][0, r_idx, ...])

                if r_idx < model_cfg.data.input_images + 1:
                    psnr_all_renders_cond.append(psnr)
                    ssim_all_renders_cond.append(ssim)
                    lpips_all_renders_cond.append(lpips)
                else:
                    psnr_all_renders_novel.append(psnr)
                    ssim_all_renders_novel.append(ssim)
                    lpips_all_renders_novel.append(lpips)

        if len(psnr_all_renders_cond) > 0:
            psnr_all_examples_cond.append(sum(psnr_all_renders_cond) / len(psnr_all_renders_cond))
            ssim_all_examples_cond.append(sum(ssim_all_renders_cond) / len(ssim_all_renders_cond))
            lpips_all_examples_cond.append(sum(lpips_all_renders_cond) / len(lpips_all_renders_cond))

        if len(psnr_all_renders_novel) > 0:
            psnr_all_examples_novel.append(sum(psnr_all_renders_novel) / len(psnr_all_renders_novel))
            ssim_all_examples_novel.append(sum(ssim_all_renders_novel) / len(ssim_all_renders_novel))
            lpips_all_examples_novel.append(sum(lpips_all_renders_novel) / len(lpips_all_renders_novel))

        if len(psnr_all_renders_sherf_cond) > 0:
            psnr_all_examples_sherf_cond.append(sum(psnr_all_renders_sherf_cond)/len(psnr_all_renders_sherf_cond))
        
        if len(psnr_all_renders_sherf_novel) > 0:
            psnr_all_examples_sherf_novel.append(sum(psnr_all_renders_sherf_novel)/len(psnr_all_renders_sherf_novel))

        with open("scores.txt", "a+") as f:
            f.write("{}_".format(d_idx) + example_id + \
                    " " + str(psnr_all_examples_novel[-1]) + \
                    " " + str(ssim_all_examples_novel[-1]) + \
                    " " + str(lpips_all_examples_novel[-1]) + "\n")

    psnr_all_examples_cond = [i for i in psnr_all_examples_cond if i != torch.inf]
    psnr_all_examples_novel = [i for i in psnr_all_examples_novel if i != torch.inf]

    scores = {"PSNR_cond": sum(psnr_all_examples_cond) / len(psnr_all_examples_cond),
              "SSIM_cond": sum(ssim_all_examples_cond) / len(ssim_all_examples_cond),
              "LPIPS_cond": sum(lpips_all_examples_cond) / len(lpips_all_examples_cond),
              "PSNR_novel": sum(psnr_all_examples_novel) / len(psnr_all_examples_novel),
              "SSIM_novel": sum(ssim_all_examples_novel) / len(ssim_all_examples_novel),
              "LPIPS_novel": sum(lpips_all_examples_novel) / len(lpips_all_examples_novel)}

    if len(psnr_all_examples_sherf_cond) > 0:
        scores["PSNR_cond_sherf"] = sum(psnr_all_examples_sherf_cond)/len(psnr_all_examples_sherf_cond)
    
    if len(psnr_all_renders_sherf_novel) > 0:
        scores["PSNR_novel_sherf"] = sum(psnr_all_renders_sherf_novel) / len(psnr_all_renders_sherf_novel)
    
    if len(abs_mpjpe) > 0:
        abs_mpjpe = torch.cat(abs_mpjpe)
        scores["ABS_MPJPE"] = (sum(abs_mpjpe) / len(abs_mpjpe)).item()

    return scores

@torch.no_grad()
def main(experiment_path, device_idx, split='val'):
    
    # set device and random seed
    device = torch.device("cuda:{}".format(device_idx))
    torch.cuda.set_device(device)

    # load cfg
    training_cfg = OmegaConf.load(os.path.join(experiment_path, ".hydra", "config.yaml"))

    # load model
    model = load_hmr_predictor(training_cfg)

    ckpt_loaded = torch.load(os.path.join(experiment_path, "model_latest.pth"), map_location=device)
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model = model.to(device)
    model.eval()
    print('Loaded model!')

    # instantiate dataset loader
    dataset = get_dataset(training_cfg, split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            persistent_workers=True, pin_memory=True, num_workers=10)
    
    scores = evaluate_dataset(model, dataloader, device, training_cfg)
    print(scores)
    return scores

if __name__ == "__main__":

    experiment_path = sys.argv[1]
    split = 'test' 
    scores = main(experiment_path, 0, split=split)
    with open(os.path.join(experiment_path, 
                           "{}_scores.json".format(split)), "w+") as f:
        json.dump(scores, f, indent=4)