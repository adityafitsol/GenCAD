import os
import numpy as np
import torch
import glob
from model import GaussianDiffusion1D, ResNetDiffusion, VanillaCADTransformer, CLIP, ResNetImageEncoder, GenCADClipAdapter
from utils import process_image, logits2vec
from config import ConfigAE
from cadlib.macro import EOS_IDX, MAX_TOTAL_LEN
from cadlib.visualize import vec2CADsolid
from OCC.Extend.DataExchange import write_stl_file

class GenCADService:
    def __init__(self, device_num=0):
        self.device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
        self.phase = "test"
        
        # Checkpoints
        self.cad_ckpt_path = "model/ckpt/ae_ckpt_epoch1000.pth"
        self.clip_ckpt_path = "model/ckpt/ccip_sketch_ckpt_epoch300.pth"
        self.diffusion_ckpt_path = 'model/ckpt/sketch_cond_diffusion_ckpt_epoch1000000.pt'

        # Model params
        self.resnet_params = {
            "d_in": 256, "n_blocks": 10, "d_main": 2048, "d_hidden": 2048, 
            "dropout_first": 0.1, "dropout_second": 0.1, "d_out": 256
        }

        self._load_models()

    def _load_models(self):
        print("Loading models...")
        
        # 1. Load diffusion prior model
        self.diffusion_model = ResNetDiffusion(
            d_in=self.resnet_params["d_in"], n_blocks=self.resnet_params["n_blocks"], 
            d_main=self.resnet_params["d_main"], d_hidden=self.resnet_params["d_hidden"], 
            dropout_first=self.resnet_params["dropout_first"], 
            dropout_second=self.resnet_params["dropout_second"], 
            d_out=self.resnet_params["d_out"]
        )

        self.diffusion = GaussianDiffusion1D(
            self.diffusion_model,
            z_dim=256,
            timesteps=500,
            objective='pred_x0', 
            auto_normalize=False
        )

        ckpt = torch.load(self.diffusion_ckpt_path, map_location="cpu")
        self.diffusion.load_state_dict(ckpt['model'])
        self.diffusion = self.diffusion.to(self.device)
        self.diffusion.eval()
        print("Diffusion checkpoint loaded.")

        # 2. Load CCIP model
        cfg_cad = ConfigAE(phase=self.phase, device=self.device, overwrite=False)
        cad_encoder = VanillaCADTransformer(cfg_cad)

        vision_network = "resnet-18"
        image_encoder = ResNetImageEncoder(network=vision_network)

        self.clip = CLIP(image_encoder=image_encoder, cad_encoder=cad_encoder, dim_latent=256)
        clip_checkpoint = torch.load(self.clip_ckpt_path, map_location='cpu')
        self.clip.load_state_dict(clip_checkpoint['model_state_dict'])
        self.clip.eval()
        
        self.clip_adapter = GenCADClipAdapter(clip=self.clip).to(self.device)
        print("CCIP checkpoint loaded.")

        # 3. Load CAD decoder model
        config = ConfigAE(exp_name="inference_service", phase="test", batch_size=1, device=self.device, overwrite=False)
        self.cad_decoder = VanillaCADTransformer(config).to(self.device)

        cad_ckpt = torch.load(self.cad_ckpt_path, map_location=self.device)
        self.cad_decoder.load_state_dict(cad_ckpt['model_state_dict'])
        self.cad_decoder.eval()
        print("CAD checkpoint loaded.")

    def generate_stl(self, img_path, export_path):
        img = process_image(img_path).to(self.device)

        with torch.no_grad():
            image_embed = self.clip_adapter.embed_image(img, normalization=False)
            latent = self.diffusion.sample(cond=image_embed)
            latent = latent.unsqueeze(0) # (1, 256) --> (1, 1, 256)

            # Decode
            outputs = self.cad_decoder(None, None, z=latent, return_tgt=False)
            batch_out_vec = logits2vec(outputs, device=self.device)
            
            # Begin loop vec
            begin_loop_vec = np.full((batch_out_vec.shape[0], 1, batch_out_vec.shape[2]), -1, dtype=np.int64)
            begin_loop_vec[:, :, 0] = 4
            auto_batch_out_vec = np.concatenate([begin_loop_vec, batch_out_vec], axis=1)[:, :MAX_TOTAL_LEN, :]

        out_vec = auto_batch_out_vec[0]
        out_command = out_vec[:, 0]

        try:
            seq_len = out_command.tolist().index(EOS_IDX)
            cad_vec = out_vec[:seq_len]
            
            # Vector to CAD
            out_vec_float = cad_vec.astype(float)
            shape = vec2CADsolid(out_vec_float)
            
            # Export STL
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            write_stl_file(shape, export_path, mode="binary", linear_deflection=0.5, angular_deflection=0.3)
            return True
        except Exception as e:
            print(f"Error generating CAD: {e}")
            return False

if __name__ == "__main__":
    # Simple test logic
    service = GenCADService()
    test_img = "data/test_images/00010010_0.png"
    if os.path.exists(test_img):
        service.generate_stl(test_img, "results/test_output.stl")
        print("Test generation complete.")
    else:
        print(f"Test image not found: {test_img}")
