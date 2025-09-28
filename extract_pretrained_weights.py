import torch
from retnext.modules import RetNeXt


ckpt_path = 'ml_experiments/augmentation_cubic_boltzmann/final_all/lightning_logs/version_0/checkpoints/best.ckpt'
ckpt = torch.load(ckpt_path)

params = ckpt['state_dict']
backbone_params = {
        k.removeprefix('model.backbone.'): v for k, v in params.items()
        if k.startswith('model.backbone')
        }

# Check that can be loaded, convert to CPU and saved them.
backbone = RetNeXt().backbone
backbone.load_state_dict(backbone_params)
backbone.cpu()
torch.save(backbone.state_dict(), 'pretrained_weights/retnext_cubic_boltzmann_final_all.pt')
