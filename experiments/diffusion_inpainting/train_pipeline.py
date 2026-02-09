#!/usr/bin/env python
# coding=utf-8
#!/usr/bin/env python3
# train_pipeline.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import argparse
from typing import List, Dict, Any
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from peft import LoraConfig


# ----------------------------
# Dataset: expects jsonl rows with fields:
# {"source":"path/to/src.png","target":"path/to/tgt.png","mask":"path/to/mask.png","prompt":"..."}
# mask: white(255) = editable region
# ----------------------------
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

CROP = (650, 0, 650 + 512, 512)  # (x1,y1,x2,y2)

class InpaintPairsDataset(Dataset):
    """
    Reads JSONL rows with:
      {"source": "...full_or_any_path...", "target": "...", "mask": "...", "prompt": "..."}
    Does NOT change JSONL; crops internally to 512x512 at CROP.
    """
    def __init__(self, jsonl_path: str):
        self.items = [json.loads(l) for l in open(jsonl_path, "r", encoding="utf-8") if l.strip()]

    def __len__(self):
        return len(self.items)

    @staticmethod
    def pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.uint8)  # (H,W,3)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t * 2.0 - 1.0

    @staticmethod
    def pil_mask_to_tensor(mask: Image.Image) -> torch.Tensor:
        arr = np.array(mask, dtype=np.uint8)  # (H,W)
        t = torch.from_numpy(arr).float() / 255.0
        return (t > 0.5).float().unsqueeze(0)

    def __getitem__(self, i):
        it = self.items[i]
        src_full = Image.open(it["source"]).convert("RGB")
        tgt_full = Image.open(it["target"]).convert("RGB")
        msk_full = Image.open(it["mask"]).convert("L")

        # crop fixed window
        src = src_full.crop(CROP)  # 512x512
        tgt = tgt_full.crop(CROP)
        msk = msk_full.crop(CROP)

        return {
            "source": self.pil_to_tensor(src),
            "target": self.pil_to_tensor(tgt),
            "mask": self.pil_mask_to_tensor(msk),
            "prompt": it["prompt"], # + ". Keep the same font, size, alignment, and ink color. Do not change any other text or background.",
        }


#class InpaintPairsDataset(Dataset):
#    def __init__(self, jsonl_path: str, size: int = 512):
#        self.items = [json.loads(l) for l in open(jsonl_path, "r", encoding="utf-8") if l.strip()]
#        self.size = size
#
#    def _load_img(self, p, is_mask=False):
#        im = Image.open(p)
#        if is_mask:
#            im = im.convert("L").resize((self.size, self.size), Image.NEAREST)
#        else:
#            im = im.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
#        return im
#
#    def __len__(self):
#        return len(self.items)
#
#    def __getitem__(self, i: int) -> Dict[str, Any]:
#        it = self.items[i]
#        src = self._load_img(it["source"], is_mask=False)
#        tgt = self._load_img(it["target"], is_mask=False)
#        msk = self._load_img(it["mask"], is_mask=True)
#        return {
#            "source": self.pil_to_tensor(src),   # [-1,1] float32
#            "target": self.pil_to_tensor(tgt),
#            "mask": self.pil_mask_to_tensor(msk),# {0,1} float32
#            "prompt": it.get("prompt","")
#        }
#
#    @staticmethod
#    def pil_to_tensor(img: Image.Image):
#        arr = np.array(img, dtype=np.uint8)  # (H,W,3), writable
#        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
#        return t * 2.0 - 1.0
#    
#    @staticmethod
#    def pil_mask_to_tensor(img: Image.Image):
#        arr = np.array(img, dtype=np.uint8)  # (H,W)
#        t = torch.from_numpy(arr).float() / 255.0
#        return (t > 0.5).float().unsqueeze(0)


    #@staticmethod
    #def pil_to_tensor(img: Image.Image):
    #    arr = (torch.frombuffer(img.tobytes(), dtype=torch.uint8).clone()
    #    #arr = (torch.frombuffer(img.tobytes(), dtype=torch.uint8)
    #           .view(img.size[1], img.size[0], 3).permute(2,0,1).float() / 255.0)
    #    return arr * 2.0 - 1.0

    #@staticmethod
    #def pil_mask_to_tensor(img: Image.Image):
    #    arr = (torch.frombuffer(img.tobytes(), dtype=torch.uint8).clone()
    #    #arr = (torch.frombuffer(img.tobytes(), dtype=torch.uint8)
    #           .view(img.size[1], img.size[0]).float() / 255.0)
    #    return (arr > 0.5).float().unsqueeze(0)


@torch.no_grad()
def encode_prompts(pipe, prompts: List[str], device: str):
    tok = pipe.tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    return pipe.text_encoder(tok.input_ids.to(device))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--model_id", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use FP16 autocast for UNet forward (keeps params in FP32)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Force model weights to FP32 to keep gradients/params FP32 (GradScaler friendly)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)

    # set modules
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.unet.train()

    # Freeze base weights
    for p in pipe.vae.parameters(): p.requires_grad_(False)
    for p in pipe.text_encoder.parameters(): p.requires_grad_(False)
    for p in pipe.unet.parameters(): p.requires_grad_(False)

    # Add PEFT LoRA adapters to UNet (target SD1.5 attention layers)
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    pipe.unet.add_adapter(lora_config)

    # Make sure UNet (and LoRA params) remain FP32
    pipe.unet.to(device=device, dtype=torch.float32)

    # Optimizer on trainable params (LoRA adapters only)
    trainable = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    # Scheduler (DDPMScheduler to add noise)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    ds = InpaintPairsDataset(args.train_jsonl)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # autocast dtype for forward
    amp_dtype = torch.float16 if (args.mixed_precision and device == "cuda") else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype==torch.float16))

    def encode_vae(x: torch.Tensor):
        # ensure fp32 for VAE
        x = x.to(device=device, dtype=torch.float32)
        lat = pipe.vae.encode(x).latent_dist.sample()
        return lat * 0.18215

    global_step = 0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        for step, batch in enumerate(dl):
            # Keep inputs in FP32 (VAE expects FP32 weights here)
            src = batch["source"].to(device=device, dtype=torch.float32)
            tgt = batch["target"].to(device=device, dtype=torch.float32)
            mask = batch["mask"].to(device=device, dtype=torch.float32)
            prompts = list(batch["prompt"])

            with torch.no_grad():
                prompt_embeds = encode_prompts(pipe, prompts, device)

                tgt_latents = encode_vae(tgt)
                masked_src = src * (1.0 - mask)   # remove surname region to provide context
                masked_latents = encode_vae(masked_src)

            bsz = tgt_latents.shape[0]
            noise = torch.randn_like(tgt_latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            noisy_latents = noise_scheduler.add_noise(tgt_latents, noise, timesteps)
            mask_lat = F.interpolate(mask, size=noisy_latents.shape[-2:], mode="nearest")

            # UNet input for inpainting: [noisy_latents (4), mask (1), masked_image_latents (4)] -> 9 channels
            unet_in = torch.cat([noisy_latents, mask_lat, masked_latents], dim=1)

            with torch.amp.autocast("cuda", enabled=(amp_dtype==torch.float16), dtype=amp_dtype):
                noise_pred = pipe.unet(unet_in, timesteps, encoder_hidden_states=prompt_embeds).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if global_step % 50 == 0:
                print(f"epoch={epoch} step={global_step} loss={loss.item()*args.grad_accum:.4f}")

        # Save adapter weights / unet adapter in PEFT format at each epoch
        save_dir = os.path.join(args.out_dir, f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        # save the UNet adapter parameters / state
        pipe.unet.save_pretrained(save_dir)
        print("Saved adapter to", save_dir)

    # save final
    pipe.unet.save_pretrained(args.out_dir)
    print("Done. Final adapter saved to:", args.out_dir)


if __name__ == "__main__":
    main()

