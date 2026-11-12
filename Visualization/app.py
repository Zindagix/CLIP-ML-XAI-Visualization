import numpy as np
import plotly.graph_objects as go
import streamlit as st

import clip_viz as viz

st.set_page_config(page_title="CLIP retrieval visualization", layout="wide")

PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
]
LABELS = {"zero-shot": "zero-shot (baseline)", "fine-tuned": "fine-tuned"}
PLOT_CONFIG = {"scrollZoom": True, "displaylogo": False}


@st.cache_resource
def load_models():
    return {tag: viz.get_model(tag) for tag in viz.TAGS}


@st.cache_data
def load_candidates(image_idx):
    texts, _ = viz.caption_candidates(image_idx)
    return texts


@st.cache_data
def load_map():
    return viz.build_semantic_assets()


@st.cache_data(show_spinner=False)
def cam_bundle(image_idx, target_idx, grid):
    texts = load_candidates(image_idx)
    return {
        tag: viz.gradcam_panel(MODELS[tag], image_idx, texts, target_idx, grid, label=LABELS[tag])
        for tag in viz.TAGS
    }


@st.cache_data(show_spinner=False)
def fovex_bundle(image_idx, target_idx, grid):
    texts = load_candidates(image_idx)
    return {
        tag: viz.fovex_panel(MODELS[tag], image_idx, texts, target_idx, grid, label=LABELS[tag])
        for tag in viz.TAGS
    }


data = viz.load_data()
MODELS = load_models()
test_idx = data["splits"]["test"]
caps_test = viz.cap_indices(data, test_idx)
test_super = data["supercategory"][test_idx]
caption_texts = np.array(data["captions"], dtype=object)


def image_label(image_idx):
    first = data["captions_by_image"][data["image_ids"][image_idx]][0]
    return f"{data['image_ids'][image_idx]} | {first[:58]}"


def map_figure(tag, colour_mode, image_idx):
    # joint map, selected image and nearest captions
    assets = load_map()[tag]
    img_xy = assets["img_xy"]
    txt_xy = assets["txt_xy"]
    img_emb, txt_emb = viz.get_embeddings(tag)
    local = int(np.searchsorted(test_idx, image_idx))
    own = txt_xy[5 * local : 5 * local + 5]
    sims = txt_emb[caps_test] @ img_emb[image_idx]
    own_text = [
        f"{sims[5 * local + i]:.3f} | {viz.short(caption_texts[c], 90)}"
        for i, c in enumerate(caps_test[5 * local : 5 * local + 5])
    ]
    near = np.argsort(-sims)[:8]
    ring = "#ffffff" if st.context.theme.type == "dark" else "#111111"

    fig = go.Figure()
    legend_specs = [
        ("captions", dict(size=7, color="#bdbdbd")),
        ("captions of the selected image", dict(size=10, color="#d62728", symbol="star")),
        ("nearest captions", dict(size=9, color="#ff7f0e", symbol="diamond")),
        ("selected image", dict(size=12, color=ring, symbol="circle-open", line=dict(width=2, color=ring))),
    ]
    for name, marker in legend_specs:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=marker,
                name=name,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scattergl(
            x=txt_xy[:, 0],
            y=txt_xy[:, 1],
            mode="markers",
            marker=dict(size=3, color="#bdbdbd", opacity=0.55),
            text=[viz.short(c, 96) for c in caption_texts[caps_test]],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
            name="captions",
        )
    )
    if colour_mode == "supercategory":
        for i, name in enumerate(sorted(set(test_super))):
            mask = test_super == name
            fig.add_trace(
                go.Scattergl(
                    x=img_xy[mask, 0],
                    y=img_xy[mask, 1],
                    mode="markers",
                    marker=dict(size=5, color=PALETTE[i % len(PALETTE)]),
                    text=[f"image {data['image_ids'][j]}" for j in test_idx[mask]],
                    hovertemplate="%{text}<extra></extra>",
                    name=name,
                )
            )
    else:
        fig.add_trace(
            go.Scattergl(
                x=img_xy[:, 0],
                y=img_xy[:, 1],
                mode="markers",
                marker=dict(size=5, color="#1f77b4"),
                text=[f"image {data['image_ids'][j]}" for j in test_idx],
                hovertemplate="%{text}<extra></extra>",
                name="images",
            )
        )
    fig.add_trace(
        go.Scattergl(
            x=own[:, 0],
            y=own[:, 1],
            mode="markers+text",
            marker=dict(size=11, color="#d62728", symbol="star"),
            text=[f"caption {i + 1}" for i in range(len(own))],
            textposition="top center",
            textfont=dict(size=9, color="#d62728"),
            customdata=own_text,
            hovertemplate="%{customdata}<extra>caption</extra>",
            showlegend=False,
            name="captions of the selected image",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=txt_xy[near, 0],
            y=txt_xy[near, 1],
            mode="markers",
            marker=dict(size=8, color="#ff7f0e", symbol="diamond"),
            text=[
                f"{sims[i]:.3f} | {test_super[i // 5]} | image {data['image_ids'][test_idx[i // 5]]} | {viz.short(caption_texts[caps_test[i]], 72)}"
                for i in near
            ],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
            name="nearest captions",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=[img_xy[local, 0]],
            y=[img_xy[local, 1]],
            mode="markers",
            marker=dict(size=15, color=ring, symbol="circle-open", line=dict(width=2, color=ring)),
            text=[f"image {data['image_ids'][image_idx]}"],
            hovertemplate="%{text}<extra>selected image</extra>",
            showlegend=False,
            name="selected image",
        )
    )
    fig.update_layout(
        title=f"{tag} | image {data['image_ids'][image_idx]}",
        height=560,
        margin=dict(l=0, r=0, t=34, b=0),
        hovermode="closest",
        dragmode="pan",
        xaxis=dict(visible=False, showgrid=False),
        yaxis=dict(visible=False, showgrid=False),
        legend=dict(font=dict(size=9), itemsizing="constant"),
    )
    return fig


st.title("Cross-modal retrieval, semantic map and explanations")

image_idx = st.sidebar.selectbox(
    "Image", list(test_idx), format_func=image_label, index=0
)
texts = load_candidates(image_idx)
target_idx = st.sidebar.selectbox(
    "Target caption",
    range(len(texts)),
    format_func=lambda i: f"{i + 1} | {viz.short(texts[i], 88)}",
)
grid = st.sidebar.slider("Grid", 1, 6, 4)
st.sidebar.caption(viz.model_status())

tab_map, tab_cam, tab_fovex = st.tabs(["Semantic map", "Grad-CAM", "FovEx"])

with tab_map:
    view = st.radio("View", ["zero-shot", "fine-tuned", "compare"], horizontal=True)
    colour = st.radio("Colour", ["supercategory", "modality"], horizontal=True)
    if view == "compare":
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                map_figure("zero-shot", colour, image_idx),
                width="stretch",
                config=PLOT_CONFIG,
            )
        with right:
            st.plotly_chart(
                map_figure("fine-tuned", colour, image_idx),
                width="stretch",
                config=PLOT_CONFIG,
            )
    else:
        st.plotly_chart(
            map_figure(view, colour, image_idx),
            width="stretch",
            config=PLOT_CONFIG,
        )

with tab_cam:
    st.caption(f"target caption {texts[target_idx]}")
    with st.spinner("computing Grad-CAM"):
        panels = cam_bundle(image_idx, target_idx, grid)
    st.iframe(
        viz.compare_html({"a": panels["zero-shot"], "b": panels["fine-tuned"]}, grid),
        height=760,
    )

with tab_fovex:
    st.caption(f"target caption {texts[target_idx]}")
    with st.spinner("running FovEx"):
        panels = fovex_bundle(image_idx, target_idx, grid)
    st.iframe(
        viz.compare_html({"a": panels["zero-shot"], "b": panels["fine-tuned"]}, grid),
        height=760,
    )
