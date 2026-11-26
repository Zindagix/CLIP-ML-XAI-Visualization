import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from transformers import CLIPModel, CLIPTokenizer

SEED = 100
CLIP_NAME = "openai/clip-vit-base-patch32"
NUM_WORKERS = 0
SIZE = 224

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"
COCO_DIR = DATA / "coco_val2017"
IMG_DIR = COCO_DIR / "val2017"
ANN_DIR = COCO_DIR / "annotations"
ASSET_DIR = Path(__file__).resolve().parent / "deploy_assets"
ASSET_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else None

MODELS = {
    "zero-shot": dict(label="Zero-shot ViT-B/32", checkpoint=None, wino_group=0.0825,
                      note="pre-trained CLIP"),
    "coco": dict(label="COCO-tuned", checkpoint=CKPT / "coco_best.pt", wino_group=0.0800,
                 note="last 4 blocks, lr 5e-05"),
    "winoground": dict(label="Winoground WiSE-FT", checkpoint=CKPT / "wino_wise_best.pt",
                       wino_group=0.1070, note="full fine-tune, WiSE alpha 0.4"),
}


def amp():
    return torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None)


@lru_cache(maxsize=1)
def load_data():
    with open(ANN_DIR / "captions_val2017.json", encoding="utf-8") as f:
        ann = json.load(f)
    caps = {}
    for a in ann["annotations"]:
        caps.setdefault(a["image_id"], []).append(a["caption"].strip())

    image_ids = sorted(caps)
    image_paths = [IMG_DIR / f"{i:012d}.jpg" for i in image_ids]
    captions_by_image = {i: caps[i][:5] for i in image_ids}
    caption_texts = [c for i in image_ids for c in captions_by_image[i]]
    offsets = np.arange(len(image_ids) + 1, dtype=np.int64) * 5

    perm = np.random.default_rng(SEED).permutation(len(image_ids))
    test_idx = np.sort(perm[4000:5000])
    cap_test = np.concatenate([np.arange(offsets[j], offsets[j + 1]) for j in test_idx])
    owner = np.repeat(np.arange(len(test_idx)), 5)

    return dict(image_ids=image_ids, image_paths=image_paths,
                captions_by_image=captions_by_image, caption_texts=caption_texts,
                offsets=offsets, test_idx=test_idx, cap_test=cap_test, owner=owner)


MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
BICUBIC = T.InterpolationMode.BICUBIC
eval_tf = T.Compose([T.Resize(SIZE, interpolation=BICUBIC), T.CenterCrop(SIZE),
                     T.ToTensor(), T.Normalize(MEAN, STD)])

_tokenizer = None


def tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = CLIPTokenizer.from_pretrained(CLIP_NAME)
    return _tokenizer


def tokenize(texts):
    t = tokenizer()(list(texts), padding=True, truncation=True, max_length=77, return_tensors="pt")
    return t["input_ids"].to(DEVICE), t["attention_mask"].to(DEVICE)


class ImageDS(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = list(paths), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def embed_images(model, paths, bs=128):
    model.eval()
    out = []
    for x in DataLoader(ImageDS(paths, eval_tf), batch_size=bs, num_workers=NUM_WORKERS):
        with amp():
            f = model.visual_projection(model.vision_model(pixel_values=x.to(DEVICE)).pooler_output)
        out.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def embed_texts(model, texts, bs=512):
    model.eval()
    texts = list(texts)
    out = []
    for s in range(0, len(texts), bs):
        ids, mask = tokenize(texts[s:s + bs])
        with amp():
            f = model.text_projection(model.text_model(input_ids=ids, attention_mask=mask).pooler_output)
        out.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


_models = {}


def get_model(tag):
    if tag not in _models:
        model = CLIPModel.from_pretrained(CLIP_NAME, use_safetensors=True).to(DEVICE).eval()
        path = MODELS[tag]["checkpoint"]
        if path is not None:
            if not Path(path).exists():
                raise FileNotFoundError(f"missing checkpoint {Path(path).name}")
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
        for p in model.parameters():
            p.requires_grad_(False)
        _models[tag] = model
    return _models[tag]


_embeddings = {}


def get_embeddings(tag):
    if tag in _embeddings:
        return _embeddings[tag]

    path = ASSET_DIR / f"emb_{tag}.npz"
    if path.exists():
        d = np.load(path)
        _embeddings[tag] = (d["img"], d["txt"])
        return _embeddings[tag]

    data = load_data()
    model = get_model(tag)
    img = embed_images(model, [str(data["image_paths"][j]) for j in data["test_idx"]])
    txt = embed_texts(model, [data["caption_texts"][i] for i in data["cap_test"]])
    np.savez(path, img=img, txt=txt)
    _embeddings[tag] = (img, txt)
    return _embeddings[tag]


def model_info(tag):
    path = MODELS[tag]["checkpoint"]
    return dict(tag=tag, label=MODELS[tag]["label"], note=MODELS[tag]["note"],
                wino_group=MODELS[tag]["wino_group"],
                ready=path is None or Path(path).exists(),
                checkpoint="base weights" if path is None else Path(path).name)


def retrieval_metrics(img, txt, owner):
    scores = txt @ img.T
    pos = np.zeros(scores.shape, dtype=bool)
    pos[np.arange(len(txt)), owner] = True
    order = np.argsort(-scores, axis=1)
    hit = np.take_along_axis(pos, order, axis=1)
    rank_t = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, hit.shape[1])
    order_t = np.argsort(-scores.T, axis=1)
    hit_i = np.take_along_axis(pos.T, order_t, axis=1)
    rank_i = np.where(hit_i.any(axis=1), hit_i.argmax(axis=1) + 1, hit_i.shape[1])
    out = {}
    for k in (1, 5, 10):
        out[f"t2i R@{k}"] = float((rank_t <= k).mean())
        out[f"i2t R@{k}"] = float((rank_i <= k).mean())
    out["t2i medR"] = float(np.median(rank_t))
    out["i2t medR"] = float(np.median(rank_i))
    return out


def evaluate(tag):
    data = load_data()
    img, txt = get_embeddings(tag)
    return dict(tag=tag, **retrieval_metrics(img, txt, data["owner"]))


def text_to_image(query, tag, k=8):
    data = load_data()
    q = embed_texts(get_model(tag), [query])[0]
    img, _ = get_embeddings(tag)
    scores = img @ q
    order = np.argsort(-scores)[:k]
    out = []
    for j in order:
        image_id = int(data["image_ids"][data["test_idx"][j]])
        out.append(dict(image_id=image_id, score=float(scores[j]),
                        caption=data["captions_by_image"][image_id][0]))
    return out


def image_to_text(image_id, tag, k=5):
    data = load_data()
    local = {int(data["image_ids"][j]): p for p, j in enumerate(data["test_idx"])}
    if int(image_id) not in local:
        raise KeyError(image_id)
    p = local[int(image_id)]
    img, txt = get_embeddings(tag)
    scores = txt @ img[p]
    order = np.argsort(-scores)[:k]
    out = []
    for o in order:
        out.append(dict(caption=data["caption_texts"][data["cap_test"][o]],
                        score=float(scores[o]),
                        image_id=int(data["image_ids"][data["test_idx"][o // 5]])))
    return out


def test_images(limit=200):
    data = load_data()
    return [dict(image_id=int(data["image_ids"][j]),
                 caption=data["captions_by_image"][data["image_ids"][j]][0])
            for j in data["test_idx"][:limit]]


def image_path(image_id):
    return IMG_DIR / f"{int(image_id):012d}.jpg"
