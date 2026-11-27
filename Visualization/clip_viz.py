import base64
import io
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from transformers import CLIPConfig, CLIPModel, CLIPProcessor
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

SEED = 100
SIZE = 224
CLIP_NAME = "openai/clip-vit-base-patch32"
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
TAGS = ("zero-shot", "fine-tuned")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIZ_DIR = Path(__file__).resolve().parent
ROOT = VIZ_DIR.parent
COCO_DIR = ROOT / "data" / "coco_val2017"
IMG_DIR = COCO_DIR / "val2017"
ANN_DIR = COCO_DIR / "annotations"
CACHE_DIR = ROOT / "cache"
CKPT = ROOT / "checkpoints" / "coco_best.pt"
ASSET_DIR = VIZ_DIR / "viz_assets"


@lru_cache(maxsize=1)
def load_data():
    with open(ANN_DIR / "captions_val2017.json", encoding="utf-8") as handle:
        caption_data = json.load(handle)
    with open(ANN_DIR / "instances_val2017.json", encoding="utf-8") as handle:
        instance_data = json.load(handle)

    captions_by_image = {}
    for ann in caption_data["annotations"]:
        bucket = captions_by_image.setdefault(ann["image_id"], [])
        if len(bucket) < 5:
            bucket.append(ann["caption"])
    image_ids = sorted(captions_by_image)
    image_paths = [IMG_DIR / f"{image_id:012d}.jpg" for image_id in image_ids]

    category_super = {c["id"]: c["supercategory"] for c in instance_data["categories"]}
    counts = {}
    for ann in instance_data["annotations"]:
        bucket = counts.setdefault(ann["image_id"], Counter())
        bucket[ann["category_id"]] += 1

    supers = []
    for image_id in image_ids:
        bucket = counts.get(image_id)
        supers.append("unlabeled" if not bucket else category_super[bucket.most_common(1)[0][0]])

    captions = [text for image_id in image_ids for text in captions_by_image[image_id]]
    offsets = np.zeros(len(image_ids) + 1, dtype=np.int64)
    for j in range(len(image_ids)):
        offsets[j + 1] = offsets[j] + len(captions_by_image[image_ids[j]])
    assert np.all(np.diff(offsets) == 5)
    owner = np.repeat(np.arange(len(image_ids)), 5)

    perm = np.random.default_rng(SEED).permutation(len(image_ids))
    splits = {
        "train": np.sort(perm[:3000]),
        "val": np.sort(perm[3000:4000]),
        "test": np.sort(perm[4000:5000]),
    }
    return dict(
        image_ids=image_ids,
        image_paths=image_paths,
        captions_by_image=captions_by_image,
        captions=captions,
        offsets=offsets,
        owner=owner,
        supercategory=np.array(supers),
        splits=splits,
    )


def cap_indices(data, idx):
    offsets = data["offsets"]
    return np.concatenate([np.arange(offsets[j], offsets[j + 1]) for j in idx])


@lru_cache(maxsize=1)
def get_processor():
    return CLIPProcessor.from_pretrained(CLIP_NAME)


@lru_cache(maxsize=1)
def load_delta():
    if CKPT.exists():
        return torch.load(CKPT, map_location="cpu", weights_only=True)
    return {}


@lru_cache(maxsize=1)
def _stamp():
    return int(CKPT.stat().st_mtime) if CKPT.exists() else 0


def build_model(delta=None):
    # config and bin weights, safetensors absent from the cached revision
    model = CLIPModel(CLIPConfig.from_pretrained(CLIP_NAME))
    weights = hf_hub_download(CLIP_NAME, "pytorch_model.bin", local_files_only=True)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    if delta:
        model.load_state_dict(delta, strict=False)
    model = model.to(DEVICE).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


_MODELS = {}


def get_model(tag):
    # lazy model cache
    if tag not in _MODELS:
        _MODELS[tag] = build_model(load_delta() if tag == "fine-tuned" else None)
    return _MODELS[tag]


def model_status():
    if CKPT.exists():
        return f"checkpoint {CKPT.name} with {len(load_delta())} tensors"
    return "checkpoint missing, zero shot weights used"


def l2norm(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def encode_images(model, paths, batch_size=64):
    out = []
    for start in range(0, len(paths), batch_size):
        batch = [Image.open(p).convert("RGB") for p in paths[start : start + batch_size]]
        inputs = get_processor()(images=batch, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_features(pixel_values=inputs["pixel_values"].to(DEVICE))
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out)


def encode_texts(model, texts, batch_size=256):
    out = []
    for start in range(0, len(texts), batch_size):
        inputs = get_processor()(
            text=texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            emb = model.get_text_features(**inputs)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out)


def embed_texts(model, texts):
    return l2norm(encode_texts(model, texts))


_EMBEDDINGS = {}


def get_embeddings(tag):
    # cached projections, l2 normalised
    if tag in _EMBEDDINGS:
        return _EMBEDDINGS[tag]
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / f"emb_{tag.replace('-', '_')}.npz"
    stamp = _stamp() if tag == "fine-tuned" else 0
    if path.exists():
        stored = np.load(path)
        if int(np.asarray(stored["stamp"]).reshape(-1)[0]) == stamp:
            _EMBEDDINGS[tag] = (stored["img"], stored["txt"])
            return _EMBEDDINGS[tag]
    data = load_data()
    cached = CACHE_DIR / "ViT-B-32.npz"
    if tag == "zero-shot" and cached.exists():
        with np.load(cached) as raw:
            img, txt = raw["img_cls"], raw["txt_proj"]
        print(f"cached zero shot features {img.shape}")
    else:
        model = get_model(tag)
        img = encode_images(model, data["image_paths"])
        txt = encode_texts(model, data["captions"])
        print(f"encoded {tag} images {img.shape} captions {txt.shape}")
    img, txt = l2norm(img), l2norm(txt)
    np.savez(path, img=img, txt=txt, stamp=np.array([stamp]))
    _EMBEDDINGS[tag] = (img, txt)
    return _EMBEDDINGS[tag]


def first_hit_rank(hits):
    has_hit = hits.any(axis=1)
    return np.where(has_hit, hits.argmax(axis=1) + 1, hits.shape[1])


def retrieval_metrics(img, txt, owner):
    scores = txt @ img.T
    positive = np.zeros(scores.shape, dtype=bool)
    positive[np.arange(len(txt)), owner] = True
    order_t = np.argsort(-scores, axis=1)
    rank_t = first_hit_rank(np.take_along_axis(positive, order_t, axis=1))
    order_i = np.argsort(-scores.T, axis=1)
    hits_i = np.take_along_axis(positive.T, order_i, axis=1)
    rank_i = first_hit_rank(hits_i)
    result = {}
    for k in (1, 5, 10):
        result[f"t2i R@{k}"] = float((rank_t <= k).mean())
        result[f"i2t R@{k}"] = float((rank_i <= k).mean())
    result["t2i medR"] = float(np.median(rank_t))
    result["i2t medR"] = float(np.median(rank_i))
    return result


def rank_rows(img, txt, owner):
    scores = txt @ img.T
    order = np.argsort(-scores, axis=1)
    positions = np.argmax(order == owner[:, None], axis=1)
    return scores, order, positions + 1


def uniformity(features, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(features), min(n, len(features)), replace=False)
    sub = features[idx]
    sim = sub @ sub.T
    d2 = np.clip(2 - 2 * sim, 0, None)
    off = ~np.eye(len(idx), dtype=bool)
    return float(np.log(np.exp(-2 * d2[off]).mean()))


def geometry_scores(img, txt, owner, seed=SEED):
    positive = (img[owner] * txt).sum(1)
    return dict(
        alignment=float(positive.mean()),
        image_uniformity=uniformity(img, seed=seed),
        caption_uniformity=uniformity(txt, seed=seed),
    )


def gallery_accuracy(img, txt, idx, caps, owner, sizes=(2, 4, 8, 16, 32, 64, 128, 256, 512, 1000), seed=SEED):
    scores = txt[caps] @ img[idx].T
    rng = np.random.default_rng(seed)
    positive = scores[np.arange(len(caps)), owner]
    n_gallery = scores.shape[1]
    out = []
    for n in sizes:
        ranks = np.empty(len(caps))
        for row in range(len(caps)):
            if n >= n_gallery:
                row_scores = scores[row]
            else:
                pick = rng.choice(n_gallery - 1, n - 1, replace=False)
                pick = np.where(pick >= owner[row], pick + 1, pick)
                row_scores = scores[row, np.concatenate([[owner[row]], pick])]
            ranks[row] = (row_scores > positive[row]).sum() + 1
        out.append(float((ranks == 1).mean()))
    return list(sizes), out


def knn_agreement(img, txt, idx, caps, owner, ks=(1, 5, 10, 20, 50)):
    scores = txt[caps] @ img[idx].T
    order = np.argsort(-scores, axis=1)
    hit = order == owner[:, None]
    cap_to_img = [float(hit[:, :k].any(1).mean()) for k in ks]
    order_t = np.argsort(-scores.T, axis=1)
    hit_t = owner[order_t] == np.arange(len(idx))[:, None]
    img_to_cap = [float(hit_t[:, :k].any(1).mean()) for k in ks]
    return list(ks), cap_to_img, img_to_cap


def pca_spectrum(*matrices, n=512):
    out = []
    for x in matrices:
        pca = PCA(n_components=min(n, x.shape[1]), random_state=SEED).fit(x)
        out.append(pca.explained_variance_ratio_)
    return out


def pca_embed(features):
    return PCA(n_components=2, random_state=SEED).fit_transform(features)


def isomap_embed(features, n_neighbors=15):
    return Isomap(n_components=2, n_neighbors=n_neighbors).fit_transform(features)


def tsne_embed(features, perplexity=30):
    n = features.shape[1]
    reduced = features
    if n > 50:
        reduced = PCA(n_components=50, random_state=SEED).fit_transform(features)
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=SEED,
        n_jobs=-1,
    ).fit_transform(reduced)


def purity(labels, clusters):
    total = 0
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        total += Counter(labels[mask]).most_common(1)[0][1]
    return total / len(labels)


def cluster_report(features, labels, k, method="kmeans", seed=SEED):
    # kmeans, ward agglomeration
    if method == "kmeans":
        pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(features)
    else:
        pred = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(features)
    metrics = dict(
        nmi=float(normalized_mutual_info_score(labels, pred)),
        ari=float(adjusted_rand_score(labels, pred)),
        purity=float(purity(labels, pred)),
        silhouette=float(silhouette_score(features, pred)),
    )
    return metrics, pred


def caption_cluster_consistency(img, txt, owner, k=13, seed=SEED):
    k_img = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(img)
    k_txt = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(txt)
    mapping = {}
    for cluster in np.unique(k_txt):
        mapping[cluster] = Counter(k_img[owner[k_txt == cluster]]).most_common(1)[0][0]
    agree = np.mean([mapping[c] == k_img[owner[i]] for i, c in enumerate(k_txt)])
    return dict(
        caption_agreement=float(agree),
        image_mutual_info=float(normalized_mutual_info_score(k_img[owner], k_txt)),
    )


def _asset_dict(stored):
    return {
        tag: dict(
            img_xy=stored[f"{tag.replace('-', '_')}_img"],
            txt_xy=stored[f"{tag.replace('-', '_')}_txt"],
        )
        for tag in TAGS
    }


def build_semantic_assets(force=False):
    # joint projection, images and captions
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / "semantic_map.npz"
    stamps = np.array([_stamp() if tag == "fine-tuned" else 0 for tag in TAGS])
    if path.exists() and not force:
        stored = np.load(path)
        if np.array_equal(stored["stamps"], stamps):
            return _asset_dict(stored)
    data = load_data()
    idx = data["splits"]["test"]
    caps = cap_indices(data, idx)
    arrays = {}
    for tag in TAGS:
        img, txt = get_embeddings(tag)
        mixed = np.vstack([img[idx], txt[caps]])
        xy = tsne_embed(mixed)
        key = tag.replace("-", "_")
        arrays[f"{key}_img"] = xy[: len(idx)]
        arrays[f"{key}_txt"] = xy[len(idx) :]
        print(f"projected {tag} points {len(xy)}")
    np.savez(path, stamps=stamps, **arrays)
    return _asset_dict(np.load(path))


def rgb_tensor(image_idx):
    data = load_data()
    image = Image.open(data["image_paths"][image_idx]).convert("RGB")
    image = TF.resize(image, SIZE, interpolation=InterpolationMode.BICUBIC, antialias=True)
    image = TF.center_crop(image, [SIZE, SIZE])
    return TF.to_tensor(image)


def normalized(rgb):
    mean = torch.tensor(MEAN, device=rgb.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=rgb.device).view(1, 3, 1, 1)
    return (rgb - mean) / std


def caption_candidates(image_idx, k=5):
    data = load_data()
    img = get_embeddings("zero-shot")[0]
    sims = img[image_idx] @ img.T
    sims[image_idx] = -1
    near = np.argsort(-sims)[:k]
    texts = list(data["captions_by_image"][data["image_ids"][image_idx]])
    texts += [data["captions_by_image"][data["image_ids"][j]][0] for j in near]
    return texts, near


class GradCAM:
    # grad cam, input tokens of the last vit block
    def __init__(self, model):
        self.model = model
        self.acts = None
        self.active = False
        layer = model.vision_model.encoder.layers[-1]
        layer.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        hidden = inputs[0]
        if self.active and hidden.requires_grad:
            hidden.retain_grad()
            self.acts = hidden

    def __call__(self, pixels, text_emb):
        x = pixels.detach().clone().requires_grad_(True)
        self.active = True
        pooled = self.model.vision_model(pixel_values=x).pooler_output
        self.active = False
        img = F.normalize(self.model.visual_projection(pooled), dim=-1)
        score = (img * text_emb).sum()
        score.backward()
        acts = self.acts[:, 1:, :].detach()
        grads = self.acts.grad[:, 1:, :]
        cam = (grads * acts).sum(-1)
        grid = int(round(cam.shape[1] ** 0.5))
        cam = cam.reshape(1, 1, grid, grid)
        cam = F.interpolate(cam, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


def gradcam_for(model):
    if not hasattr(model, "_gradcam"):
        model._gradcam = GradCAM(model)
    return model._gradcam


def heat_rgb(heat):
    colors = plt.get_cmap("jet")(np.clip(heat, 0, 1))[..., :3] * 255
    return np.clip(colors, 0, 255).astype(np.uint8)


def overlay_heatmap(rgb_uint8, heat, alpha=0.5):
    height, width = rgb_uint8.shape[:2]
    resized = np.array(Image.fromarray(heat.astype(np.float32), mode="F").resize((width, height), Image.BILINEAR))
    colors = heat_rgb(resized).astype(np.float32)
    out = (1 - alpha) * rgb_uint8.astype(np.float32) + alpha * colors
    return np.clip(out, 0, 255).astype(np.uint8)


def cam_overlay(rgb, cam, alpha=0.5):
    base = (rgb.permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return overlay_heatmap(base, cam, alpha)


def draw_scanpath(rgb_uint8, points):
    image = Image.fromarray(rgb_uint8).copy()
    draw = ImageDraw.Draw(image)
    pix = [
        ((float(p[1]) + 1) * 0.5 * (SIZE - 1), (float(p[0]) + 1) * 0.5 * (SIZE - 1))
        for p in points
    ]
    if len(pix) > 1:
        draw.line(pix, fill=(255, 220, 0), width=2)
    for i, (x, y) in enumerate(pix):
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(255, 255, 0), width=2)
        draw.text((x + 6, y - 6), str(i + 1), fill=(255, 255, 0))
    return np.array(image)


def draw_cross(rgb_uint8, point):
    image = Image.fromarray(rgb_uint8).copy()
    draw = ImageDraw.Draw(image)
    x = (float(point[1]) + 1) * 0.5 * (SIZE - 1)
    y = (float(point[0]) + 1) * 0.5 * (SIZE - 1)
    draw.line([(x - 9, y), (x + 9, y)], fill=(255, 255, 0), width=2)
    draw.line([(x, y - 9), (x, y + 9)], fill=(255, 255, 0), width=2)
    draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(255, 255, 0), width=1)
    return np.array(image)


def to_uri(array, quality=85):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def create_grid(batch_size, size, device):
    t = torch.linspace(-1.0, 1.0, size, device=device)
    xa, ya = torch.meshgrid(t, t, indexing="ij")
    xa = xa.view(1, size, size).repeat(batch_size, 1, 1)
    ya = ya.view(1, size, size).repeat(batch_size, 1, 1)
    return xa, ya


def calc_gaussian(a, std_dev, image_size, positions, device):
    batch_size = positions.shape[0]
    xa, ya = create_grid(batch_size, image_size, device)
    xa = xa - positions[:, 0].reshape(-1, 1, 1)
    ya = ya - positions[:, 1].reshape(-1, 1, 1)
    g = a * torch.exp(-(xa**2 + ya**2) / std_dev)
    return g.view(batch_size, 1, image_size, image_size)


def gaussian_window(window_size, sigma, device):
    x = torch.arange(window_size, device=device, dtype=torch.float32)
    gauss = torch.exp(-((x - window_size // 2) ** 2) / (2.0 * sigma**2))
    return gauss / gauss.sum()


def calculate_blur(images, blur_filter_size, blur_sigma):
    device = images.device
    g1d = gaussian_window(blur_filter_size, blur_sigma, device).view(-1, 1)
    g2d = g1d.mm(g1d.t()).view(1, 1, blur_filter_size, blur_filter_size)
    window = g2d.expand(images.shape[1], 1, blur_filter_size, blur_filter_size).contiguous()
    pad = nn.ReflectionPad2d(blur_filter_size // 2)
    blurred = F.conv2d(pad(images), window, groups=images.shape[1])
    return blurred - images


def get_foveation(foveation_aggregation, foveation_sigma, image_size, positions, device):
    return calc_gaussian(foveation_aggregation, foveation_sigma, image_size, positions, device)


def normalize_map(x, eps=1e-8):
    lo = x.amin(dim=(2, 3), keepdim=True)
    hi = x.amax(dim=(2, 3), keepdim=True)
    return (x - lo) / (hi - lo + eps)


def get_heat_maps(foveation_sigma, image_size, scanpaths, forgetting_factor, device, normalization):
    batch_size = scanpaths.shape[0]
    heat_map = torch.zeros((batch_size, 1, image_size, image_size), device=device)
    for i in range(scanpaths.shape[1]):
        current = get_foveation(1, foveation_sigma, image_size, scanpaths[:, i, :, None, None], device)
        heat_map = forgetting_factor[i] * heat_map + current
    if normalization:
        heat_map = normalize_map(heat_map)
    return heat_map


class CaptionScorer(nn.Module):
    def __init__(self, model, text_emb, scale):
        super().__init__()
        self.model = model
        self.register_buffer("text_emb", text_emb)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.scale = float(scale)

    def forward(self, x):
        pixels = (x - self.mean) / self.std
        pooled = self.model.vision_model(pixel_values=pixels).pooler_output
        img = F.normalize(self.model.visual_projection(pooled), dim=-1)
        return self.scale * (img @ self.text_emb.T)


class FovExWrapper(nn.Module):
    def __init__(
        self,
        downstream_model,
        criterion,
        target_function,
        image_size,
        foveation_sigma,
        blur_filter_size,
        blur_sigma,
        forgetting,
        heatmap_sigma,
        heatmap_forgetting,
        device,
        foveation_aggregation=1,
    ):
        super().__init__()
        self.downstream_model = downstream_model
        self.criterion = criterion
        self.target_function = target_function
        self.image_size = image_size
        self.foveation_sigma = foveation_sigma
        self.blur_filter_size = blur_filter_size
        self.blur_sigma = blur_sigma
        self.forgetting = forgetting
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_forgetting = heatmap_forgetting
        self.foveation_aggregation = foveation_aggregation
        self.device = device
        self.internal_representation = None
        self.ones = None
        self.blur = None

    def forward(self, x, foveation_positions):
        foveation_area = get_foveation(
            self.foveation_aggregation,
            self.foveation_sigma,
            self.image_size,
            foveation_positions,
            x.device,
        )
        current = self.internal_representation + foveation_area
        blurring_mask = torch.clamp(self.ones - current, 0.0, 1.0)
        return self.downstream_model(x + self.blur * blurring_mask)

    def initialize_scanpath_generation(self, x, batch_size):
        self.internal_representation = torch.zeros(
            (batch_size, 1, self.image_size, self.image_size), device=x.device
        )
        self.ones = torch.ones(
            (batch_size, 1, self.image_size, self.image_size), device=x.device
        )
        self.blur = calculate_blur(x, self.blur_filter_size, self.blur_sigma)

    def run_optimization(self, x, labels, scanpath_length, opt_iterations, learning_rate, random_restarts=False):
        device = x.device
        batch_size = x.size(0)
        targets = self.target_function(x, labels)
        self.initialize_scanpath_generation(x, batch_size)
        scanpath = []
        loss_history = []
        for _ in range(scanpath_length):
            foveation_pos = torch.zeros((batch_size, 2, 1, 1), device=device, requires_grad=True)
            best_pos = torch.zeros((batch_size, 2, 1, 1), device=device)
            best_loss = torch.full((batch_size,), float("inf"), device=device)
            for _ in range(opt_iterations):
                output = self(x, foveation_pos)
                loss = self.criterion(output, targets)
                total_loss = loss.mean()
                grad = torch.autograd.grad(total_loss, foveation_pos)[0]
                foveation_pos.data -= torch.sign(grad) * learning_rate
                loss_value = loss.detach()
                better = loss_value < best_loss
                best_loss[better] = loss_value[better]
                best_pos[better] = foveation_pos.detach()[better]
                if torch.sum(~better) > 0:
                    if random_restarts:
                        foveation_pos.data[~better] = (
                            torch.rand_like(foveation_pos.data[~better]) * 2.0 - 1.0
                        )
                    else:
                        foveation_pos.data[~better] += (
                            torch.rand_like(foveation_pos.data[~better]) * learning_rate - learning_rate / 2.0
                        )
            current_mask = get_foveation(
                self.foveation_aggregation, self.foveation_sigma, self.image_size, best_pos, device
            )
            self.internal_representation = (
                self.internal_representation * self.forgetting + current_mask
            ).detach()
            scanpath.append(best_pos.detach())
            loss_history.append(loss_value)
        return torch.stack(scanpath, 1).squeeze(), torch.stack(loss_history, 1).squeeze(), self.internal_representation

    def generate_explanation(
        self,
        x,
        labels,
        scanpath_length,
        opt_iterations,
        learning_rate,
        random_restarts=False,
        normalize_heatmap=True,
        seed=SEED,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        scanpaths, loss_history, internal_rep = self.run_optimization(
            x, labels, scanpath_length, opt_iterations, learning_rate, random_restarts
        )
        heatmap = get_heat_maps(
            self.heatmap_sigma,
            self.image_size,
            scanpaths[None],
            self.heatmap_forgetting,
            x.device,
            normalize_heatmap,
        )
        return heatmap, scanpaths, loss_history, internal_rep


def caption_scores_from_pixels(model, pixels, text_emb):
    pooled = model.vision_model(pixel_values=pixels).pooler_output
    img = F.normalize(model.visual_projection(pooled), dim=-1)
    return (img @ text_emb.T)[0].detach().cpu().numpy()


def caption_scores(model, rgb, text_emb):
    with torch.no_grad():
        return caption_scores_from_pixels(model, normalized(rgb), text_emb)


def scaled_probs(model, sims):
    scale = float(model.logit_scale.exp().clamp(max=100).item())
    z = scale * sims
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def fovex_run(model, image_idx, texts, target_idx, scanpath_length=10, seed=SEED):
    # fovex, sign gradient on the fixation
    rgb = rgb_tensor(image_idx).unsqueeze(0).to(DEVICE)
    text_emb = torch.tensor(embed_texts(model, texts), dtype=torch.float32, device=DEVICE)
    scorer = CaptionScorer(model, text_emb, model.logit_scale.exp().clamp(max=100).item()).to(DEVICE).eval()
    engine = FovExWrapper(
        scorer,
        nn.CrossEntropyLoss(reduction="none"),
        lambda x, y: y,
        SIZE,
        0.15,
        41,
        10.0,
        0.1,
        0.15,
        [1.0] * scanpath_length,
        DEVICE,
    )
    heat, scan, losses, _ = engine.generate_explanation(
        rgb,
        torch.tensor([target_idx], device=DEVICE),
        scanpath_length,
        20,
        0.1,
        True,
        True,
        seed,
    )
    sims = caption_scores(model, rgb, text_emb)
    return dict(
        rgb=rgb[0].detach().cpu(),
        heat=heat[0, 0].detach().cpu().numpy(),
        scan=scan.detach().cpu().numpy().reshape(-1, 2),
        sims=sims,
        probs=scaled_probs(model, sims),
        text_emb=text_emb.detach().cpu().numpy(),
        losses=losses.detach().cpu().numpy().reshape(-1),
    )


def fovex_step_views(rgb, scan, blur, sigma=0.15, beta=0.1):
    views, masks = [], []
    g = torch.zeros(1, 1, SIZE, SIZE, device=rgb.device)
    for position in scan:
        pos = torch.tensor(position, dtype=torch.float32, device=rgb.device).view(1, 2, 1, 1)
        g = g * beta + calc_gaussian(1, sigma, SIZE, pos, rgb.device)
        masks.append(g.clone())
        views.append((rgb + blur * torch.clamp(1 - g, 0, 1)).clamp(0, 1))
    return views, masks


def cell_coord(position, grid):
    row = int(np.clip(np.floor((position[0] + 1) * 0.5 * grid), 0, grid - 1))
    col = int(np.clip(np.floor((position[1] + 1) * 0.5 * grid), 0, grid - 1))
    return row, col


def short(text, n=70):
    return text if len(text) <= n else text[: n - 1] + "~"


def gradcam_panel(model, image_idx, texts, target_idx, grid, label=""):
    rgb = rgb_tensor(image_idx).unsqueeze(0).to(DEVICE)
    text_emb = torch.tensor(embed_texts(model, texts), dtype=torch.float32, device=DEVICE)
    cam = gradcam_for(model)(normalized(rgb), text_emb[target_idx])
    base = cam_overlay(rgb[0].cpu(), cam)
    base_uri = to_uri(base)
    sims = caption_scores(model, rgb, text_emb)
    probs = scaled_probs(model, sims)

    rows = np.linspace(0, SIZE, grid + 1).astype(int)
    cells = []
    for r in range(grid):
        for c in range(grid):
            y0, y1 = rows[r], rows[r + 1]
            x0, x1 = rows[c], rows[c + 1]
            share = float(cam[y0:y1, x0:x1].sum() / (cam.sum() + 1e-8))
            mean = float(cam[y0:y1, x0:x1].mean())
            removed = rgb.clone()
            removed[:, :, y0:y1, x0:x1] = 0.5
            removed_sims = caption_scores(model, removed, text_emb)
            drop = float(sims[target_idx] - removed_sims[target_idx])
            top = int(removed_sims.argmax())
            crop = F.interpolate(
                rgb[:, :, y0:y1, x0:x1], size=(160, 160), mode="bilinear", align_corners=False
            )
            crop_uri = to_uri((crop[0].permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            cells.append(
                dict(
                    img="",
                    aux=crop_uri,
                    title=f"row {r + 1}, col {c + 1}",
                    lines=[
                        f"CAM share {share:.3f}",
                        f"CAM mean {mean:.3f}",
                        f"target score drop {drop:+.4f}",
                        f"top caption after removal {top + 1}",
                        short(texts[top], 64),
                    ],
                )
            )

    default = dict(
        img="",
        aux="",
        title=f"target caption {target_idx + 1}",
        lines=[
            f"target score {sims[target_idx]:.4f}",
            f"target probability {probs[target_idx]:.3f}",
            f"top caption {int(sims.argmax()) + 1}",
            short(texts[int(sims.argmax())], 64),
        ],
    )
    return dict(label=label, default=default, cells=cells, base=base_uri)


def fovex_panel(model, image_idx, texts, target_idx, grid, scanpath_length=10, label=""):
    out = fovex_run(model, image_idx, texts, target_idx, scanpath_length)
    rgb = out["rgb"].unsqueeze(0).to(DEVICE)
    blur = calculate_blur(rgb, 41, 10.0)
    scan = out["scan"]
    text_emb = torch.tensor(out["text_emb"], dtype=torch.float32, device=DEVICE)
    views, masks = fovex_step_views(rgb, scan, blur)

    step_sims = []
    for view in views:
        with torch.no_grad():
            step_sims.append(caption_scores_from_pixels(model, normalized(view), text_emb))
    plain = (out["rgb"].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    heat_uri = to_uri(heat_rgb(out["heat"]))
    base_pred = int(out["sims"].argmax())
    full_p = float(out["probs"][target_idx])

    step_cells = {}
    for t, position in enumerate(scan):
        step_cells.setdefault(cell_coord(position, grid), []).append(t)
    final_row, final_col = cell_coord(scan[-1], grid)

    cells = []
    for r in range(grid):
        for c in range(grid):
            key = (r, c)
            if key == (final_row, final_col):
                t = len(scan) - 1
                view = views[t][0].detach().cpu().permute(1, 2, 0).numpy()
                view_u8 = (view * 255).clip(0, 255).astype(np.uint8)
                main = draw_scanpath(overlay_heatmap(view_u8, out["heat"], 0.5), scan[: t + 1])
                pred = int(step_sims[t].argmax())
                p_target = scaled_probs(model, step_sims[t])[target_idx]
                cells.append(
                    dict(
                        img=to_uri(main),
                        aux=heat_uri,
                        title=f"final fixation | row {r + 1}, col {c + 1}",
                        lines=[
                            "FovEx attribution map",
                            f"target probability {p_target:.3f} ({p_target - full_p:+.3f})",
                            f"top caption {pred + 1}",
                            short(texts[pred], 64),
                        ],
                    )
                )
            elif key in step_cells:
                t = max(step_cells[key])
                view = views[t][0].detach().cpu().permute(1, 2, 0).numpy()
                view_u8 = (view * 255).clip(0, 255).astype(np.uint8)
                main = draw_scanpath(view_u8, scan[: t + 1])
                pred = int(step_sims[t].argmax())
                p_target = scaled_probs(model, step_sims[t])[target_idx]
                cells.append(
                    dict(
                        img=to_uri(main),
                        aux=to_uri(heat_rgb(masks[t][0, 0].detach().cpu().numpy())),
                        title=f"fixation {t + 1} | row {r + 1}, col {c + 1}",
                        lines=[
                            f"step {t + 1} of {len(scan)}",
                            f"target probability {p_target:.3f} ({p_target - full_p:+.3f})",
                            f"top caption {pred + 1}",
                            short(texts[pred], 64),
                        ],
                    )
                )
            else:
                position = (2 * (r + 0.5) / grid - 1, 2 * (c + 0.5) / grid - 1)
                pos = torch.tensor(position, dtype=torch.float32, device=DEVICE).view(1, 2, 1, 1)
                mask = calc_gaussian(1, 0.15, SIZE, pos, DEVICE)
                view = (rgb + blur * torch.clamp(1 - mask, 0, 1)).clamp(0, 1)
                sims = caption_scores(model, view, text_emb)
                pred = int(sims.argmax())
                p_target = scaled_probs(model, sims)[target_idx]
                cam = gradcam_for(model)(normalized(view), text_emb[target_idx])
                view_u8 = (view[0].permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                cells.append(
                    dict(
                        img=to_uri(draw_cross(view_u8, position)),
                        aux=to_uri(cam_overlay(view[0], cam)),
                        title=f"simulated fixation | row {r + 1}, col {c + 1}",
                        lines=[
                            "foveated view, Grad-CAM",
                            f"target probability {p_target:.3f} ({p_target - full_p:+.3f})",
                            f"top caption {pred + 1}",
                            short(texts[pred], 64),
                        ],
                    )
                )

    base_uri = to_uri(draw_scanpath(plain, scan))
    default = dict(
        img="",
        aux=heat_uri,
        title=f"input | {len(scan)} fixations",
        lines=[
            f"target probability {out['probs'][target_idx]:.3f}",
            f"top caption {base_pred + 1}",
            short(texts[base_pred], 64),
        ],
    )
    return dict(label=label, default=default, cells=cells, base=base_uri)


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
 body { margin:0; padding:10px; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:#1f2328; background:#ffffff; }
 .row { display:flex; gap:28px; justify-content:center; align-items:flex-start; flex-wrap:wrap; }
 .panel { width:320px; }
 .label { font-size:13px; font-weight:600; margin-bottom:6px; }
 .frame { position:relative; width:320px; height:320px; border-radius:8px; overflow:hidden; background:#111; }
 .frame img { width:320px; height:320px; display:block; }
 .grid { position:absolute; inset:0; display:grid; }
 .cell { border:1px solid rgba(255,255,255,0.30); box-sizing:border-box; cursor:crosshair; }
 .cell:hover { background:rgba(255,255,255,0.20); border-color:#ffffff; }
 .title { font-size:12px; font-weight:600; margin-top:8px; min-height:16px; }
 .lines { font-size:11px; line-height:1.5; color:#3d444d; min-height:74px; }
 .aux { margin-top:4px; width:160px; height:160px; border-radius:6px; display:none; }
 .aux.on { display:block; }
</style>
</head>
<body>
<div class="row">
  <div class="panel">
    <div class="label" id="a_label"></div>
    <div class="frame"><img id="a_img"><div class="grid" id="a_grid"></div></div>
    <div class="title" id="a_title"></div>
    <div class="lines" id="a_lines"></div>
    <img class="aux" id="a_aux">
  </div>
  <div class="panel">
    <div class="label" id="b_label"></div>
    <div class="frame"><img id="b_img"><div class="grid" id="b_grid"></div></div>
    <div class="title" id="b_title"></div>
    <div class="lines" id="b_lines"></div>
    <img class="aux" id="b_aux">
  </div>
</div>
<script>
const DATA = __PAYLOAD__;
const N = DATA.grid;
function apply(side, panel, cell) {
  const img = document.getElementById(side + "_img");
  img.src = cell.img || panel.base;
  const aux = document.getElementById(side + "_aux");
  if (cell.aux) { aux.src = cell.aux; aux.classList.add("on"); }
  else { aux.classList.remove("on"); aux.removeAttribute("src"); }
  document.getElementById(side + "_title").textContent = cell.title || "";
  const box = document.getElementById(side + "_lines");
  box.innerHTML = "";
  (cell.lines || []).forEach(function (t) {
    const d = document.createElement("div");
    d.textContent = t;
    box.appendChild(d);
  });
}
function show(i) {
  apply("a", DATA.a, DATA.a.cells[i]);
  apply("b", DATA.b, DATA.b.cells[i]);
}
function reset() {
  apply("a", DATA.a, DATA.a.default);
  apply("b", DATA.b, DATA.b.default);
}
function build(side, panel) {
  document.getElementById(side + "_label").textContent = panel.label;
  const grid = document.getElementById(side + "_grid");
  grid.style.gridTemplateColumns = "repeat(" + N + ", 1fr)";
  grid.style.gridTemplateRows = "repeat(" + N + ", 1fr)";
  for (let i = 0; i < N * N; i++) {
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.addEventListener("mouseenter", function () { show(i); });
    grid.appendChild(cell);
  }
}
build("a", DATA.a);
build("b", DATA.b);
reset();
</script>
</body>
</html>
"""


def compare_html(panels, grid):
    # hover grid, two models
    payload = {"grid": int(grid), "a": panels["a"], "b": panels["b"]}
    body = json.dumps(payload).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__PAYLOAD__", body)
