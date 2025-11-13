import torch, cv2, numpy as np, matplotlib.pyplot as plt
from models.backbone import ResNetBackbone
from models.svf_head import SVFHead
from lib.geom import exp_map_ss, warp_landmarks, compute_confidence
from preprocessing.utils import generate_edge_bank
from config import cfg

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_once(img_path, pred_dt_np_path, atlas_landmarks_path, svf_ckpt):
    img = cv2.imread(img_path)
    assert img is not None, "image not found"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    # tensors
    img_t = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
    D_np = np.load(pred_dt_np_path)  # (H,W) or (1,H,W)
    if D_np.ndim == 2:
        D_np = D_np[np.newaxis,:,:]
    D_t = torch.from_numpy(D_np.astype(np.float32)).unsqueeze(0).to(device)  # (1,1,H,W)

    # backbone
    backbone = ResNetBackbone(name="resnet34", pretrained=False, fuse_edges=False).to(device)
    backbone.eval()
    with torch.no_grad():
        feats = backbone(img_t)
    F_feat = feats["C5"]

    # load svf
    svf = SVFHead(in_channels=F_feat.shape[1], mid_channels=128).to(device)
    svf.load_state_dict(torch.load(svf_ckpt, map_location=device))
    svf.eval()

    with torch.no_grad():
        v = svf(F_feat, D_t)                # (1,2,Hf,Wf)
        phi = exp_map_ss(v, n_steps=6)     # (1,2,Hf,Wf)

    atlas_landmarks = np.load(atlas_landmarks_path)  # (19,2) pixel coords
    atlas_t = torch.from_numpy(atlas_landmarks).unsqueeze(0).float().to(device)
    warped = warp_landmarks(atlas_t, phi, image_h=H, image_w=W).squeeze(0).cpu().numpy()

    # confidence
    conf = compute_confidence(D_t, phi, image_h=H, image_w=W).squeeze(0).cpu().numpy()

    # sample conf per landmark
    xs = (warped[:,0] / (W - 1)) * 2.0 - 1.0
    ys = (warped[:,1] / (H - 1)) * 2.0 - 1.0
    grid = np.stack([xs,ys], axis=-1)[None,:,None,:]  # (1,N,1,2)
    import torch.nn.functional as F
    grid_t = torch.from_numpy(grid).float().to(device)
    conf_t = torch.from_numpy(conf).unsqueeze(0).unsqueeze(0).to(device)
    sampled = F.grid_sample(conf_t, grid_t, align_corners=True).squeeze().cpu().numpy()
    sampled = sampled.flatten()

    # visualize
    plt.figure(figsize=(10,6))
    plt.imshow(img_rgb)
    for i,(xy,c) in enumerate(zip(warped, sampled)):
        x,y = xy
        plt.scatter(x,y, c='lime' if c>0.5 else 'red', s=35)
        plt.text(x+3,y-3, f"{i}:{c:.2f}", color='white', fontsize=8)
    plt.title("Warped landmarks (color=confidence)")
    plt.show()

    return warped, sampled

if __name__ == "__main__":
    # EXAMPLE usage - edit paths:
    img_path = "/content/landmark-detection/datasets/ISBI Dataset/Dataset/Testing/Test1/001.bmp"
    pred_dt = "/content/landmark-detection/pred_dt_example.npy"
    atlas_lm = "/content/landmark-detection/atlas_landmarks_19.npy"
    svf_ckpt = "/content/landmark-detection/checkpoints_svf/best_svf.pth"
    run_once(img_path, pred_dt, atlas_lm, svf_ckpt)
