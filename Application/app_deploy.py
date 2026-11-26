import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("CS5489_API", "http://127.0.0.1:8000")
IMG_DIR = Path(__file__).resolve().parent.parent / "data" / "coco_val2017" / "val2017"


def api(path, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=600)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600)
def models():
    return api("/models")


@st.cache_data(ttl=600)
def evaluation(tag):
    return api("/eval", tag=tag)


@st.cache_data(ttl=600)
def test_images():
    return api("/images", limit=300)


st.set_page_config(page_title="CS5489 retrieval", layout="wide")

info = models()
ready = [m for m in info if m["ready"]]
if not ready:
    st.error("no checkpoint available")
    st.stop()

labels = {m["tag"]: m["label"] for m in info}
tag = st.sidebar.selectbox("Model", [m["tag"] for m in ready], format_func=lambda t: labels[t])
k = st.sidebar.slider("Results", 1, 20, 8)
checkpoint = next(m["checkpoint"] for m in info if m["tag"] == tag) or "zero-shot weights"

st.title("Image retrieval with natural language")
st.caption(f"service {API} | checkpoint {checkpoint}")

tab_text, tab_compare, tab_image, tab_eval, tab_models = st.tabs(
    ["Text to image", "Compare models", "Image to text", "Evaluation", "Models"]
)

with tab_text:
    query = st.text_input("Query", "a man cutting a cake")
    if st.button("Search images", type="primary"):
        hits = api("/text", query=query, tag=tag, k=k)
        st.write(f"{len(hits)} results")
        cols = st.columns(4)
        for i, h in enumerate(hits):
            with cols[i % 4]:
                st.image(str(IMG_DIR / f"{h['image_id']:012d}.jpg"))
                st.write(f"{h['score']:.3f} | image {h['image_id']}")
                st.caption(h["caption"])

with tab_compare:
    query = st.text_input("Query", "a man cutting a cake", key="compare_query")
    if st.button("Compare models", type="primary"):
        for m in ready:
            st.subheader(labels[m["tag"]])
            hits = api("/text", query=query, tag=m["tag"], k=4)
            cols = st.columns(4)
            for i, h in enumerate(hits):
                with cols[i % 4]:
                    st.image(str(IMG_DIR / f"{h['image_id']:012d}.jpg"))
                    st.write(f"{h['score']:.3f} | image {h['image_id']}")

with tab_image:
    options = {f"{im['image_id']} | {im['caption'][:60]}": im["image_id"] for im in test_images()}
    choice = st.selectbox("Image", list(options))
    if st.button("Describe image", type="primary"):
        image_id = options[choice]
        hits = api("/image", image_id=image_id, tag=tag, k=k)
        st.image(str(IMG_DIR / f"{image_id:012d}.jpg"))
        for h in hits:
            st.write(f"{h['score']:.3f} | {h['caption']}")

with tab_eval:
    if st.button("Evaluate all models"):
        rows = [dict(model=labels[m["tag"]], **evaluation(m["tag"])) for m in ready]
        st.session_state["eval_rows"] = rows
    rows = st.session_state.get("eval_rows")
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("model").round(4))

with tab_models:
    df = pd.DataFrame(info).set_index("tag").rename(columns={"wino_group": "wino group (CV)"})
    st.dataframe(df)
