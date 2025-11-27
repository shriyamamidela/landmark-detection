import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.backbone import ResNetBackbone
from models.svf_head import SVFHead
from aug_dataset import AugCephDataset
from train_svf import svf_to_disp, sample_flow_at_points_local


# -----------------------------
# EXPORT PRED / GT FUNCTION
# -----------------------------
def export_predictions(test_loader, backbone, svf_head, atlas_edges, atlas_lms, device):
    all_pred = []
    all_gt = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Exporting predictions"):
            images, gt_landmarks, dt_maps, edge_maps, _ = batch

            images = images.to(device)
            gt_landmarks = gt_landmarks.to(device)
            dt_maps = dt_maps.to(device)
            edge_maps = edge_maps.to(device)

            feats = backbone(images)
            F5 = feats["C5"]

            # Downsample DT
            Hf, Wf = F5.shape[2], F5.shape[3]
            dt_small = F.interpolate(dt_maps, size=(Hf, Wf), mode="bilinear")

            # SVF + displacement
            v = svf_head(F5, dt_small)
            disp = svf_to_disp(v, steps=6)

            # Upsample to image resolution
            H, W = images.shape[2], images.shape[3]
            disp_full = F.interpolate(disp, size=(H, W))

            # Landmark prediction
            pred = atlas_lms.to(device) + sample_flow_at_points_local(
                disp_full, atlas_lms.to(device), (H, W)
            )

            all_pred.append(pred.cpu().numpy())
            all_gt.append(gt_landmarks.cpu().numpy())

    return (
        np.concatenate(all_pred, axis=0),
        np.concatenate(all_gt, axis=0)
    )


# -----------------------------
# MAIN
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # PATHS YOU MUST SET
    ckpt_path = "/dgxa_home/se22ucse250/landmark-detection-main/checkpoints/debug_svf/svf_epoch_15.pth"
    atlas_edge = "/dgxa_home/se22ucse250/landmark-detection-main/atlas/atlas_edge_map_resized.npy"
    atlas_lms = "/dgxa_home/se22ucse250/landmark-detection-main/atlas/atlas_landmarks_resized.npy"

    test1_dir = "/dgxa_home/se22ucse250/landmark-detection-main/datasets/aug_test/test1changed_1"
    test2_dir = "/dgxa_home/se22ucse250/landmark-detection-main/datasets/aug_test/test2changed_2"

    save_dir = "/dgxa_home/se22ucse250/landmark-detection-main/checkpoints/predictions"
    os.makedirs(save_dir, exist_ok=True)

    # Load datasets
    test1 = AugCephDataset(test1_dir)
    test2 = AugCephDataset(test2_dir)

    test1_loader = DataLoader(test1, batch_size=1, shuffle=False)
    test2_loader = DataLoader(test2, batch_size=1, shuffle=False)

    # Load atlas
    atlas_edges = torch.tensor(np.load(atlas_edge)).float().unsqueeze(0).unsqueeze(0).to(device)
    atlas_lms = torch.tensor(np.load(atlas_lms)).float().unsqueeze(0).to(device)

    # Load networks
    backbone = ResNetBackbone("resnet34", pretrained=True).to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    svf_head = SVFHead().to(device)

    ck = torch.load(ckpt_path, map_location=device)
    svf_head.load_state_dict(ck["svf"])
    print("Loaded checkpoint:", ckpt_path)

    # -----------------------------
    # EXPORT TEST1
    # -----------------------------
    print("Exporting Test1 predictions...")
    pred1, gt1 = export_predictions(test1_loader, backbone, svf_head, atlas_edges, atlas_lms, device)
    np.save(f"{save_dir}/test1_pred.npy", pred1)
    np.save(f"{save_dir}/test1_gt.npy", gt1)
    print("Saved Test1 predictions.")

    # -----------------------------
    # EXPORT TEST2
    # -----------------------------
    print("Exporting Test2 predictions...")
    pred2, gt2 = export_predictions(test2_loader, backbone, svf_head, atlas_edges, atlas_lms, device)
    np.save(f"{save_dir}/test2_pred.npy", pred2)
    np.save(f"{save_dir}/test2_gt.npy", gt2)
    print("Saved Test2 predictions.")

    print("\n✅ DONE — Predictions exported!")


if __name__ == "__main__":
    main()
