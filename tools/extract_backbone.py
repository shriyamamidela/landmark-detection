import torch

# path to your trained Stage-1 checkpoint
path = "/content/drive/MyDrive/atlas_checkpoints/checkpoints_resnet_edge/best_resnet_edge.pth"

ckpt = torch.load(path)

# ckpt is already a state_dict (OrderedDict)
print("Loaded checkpoint type:", type(ckpt))

state = ckpt  # raw state_dict

backbone_only = {}
for k, v in state.items():
    if k.startswith("backbone."):
        new_k = k.replace("backbone.", "")   # strip prefix
        backbone_only[new_k] = v

out_path = "/content/drive/MyDrive/atlas_checkpoints/resnet34_backbone_only.pth"
torch.save(backbone_only, out_path)

print("\nSaved cleaned backbone weights →", out_path)
print("Example keys:", list(backbone_only.keys())[:10])
