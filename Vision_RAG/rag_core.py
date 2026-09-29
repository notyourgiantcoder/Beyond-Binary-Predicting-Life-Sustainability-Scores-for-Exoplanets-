"""
Vision RAG core — the retrieval logic from vision_rag.py without any Streamlit UI,
so the main dashboard (app.py) can reuse it.

Pipeline: every image / PDF page is embedded with Cohere Embed-4; a question is
embedded the same way, the closest image is retrieved by cosine similarity, and
Gemini 2.5 Flash answers the question from that image.
"""

import base64
import io
import os

import numpy as np
import requests
from PIL import Image

EMBED_MODEL = "embed-v4.0"
ANSWER_MODEL = "gemini-2.5-flash"
MAX_PIXELS = 1568 * 1568  # max resolution sent to the embedding API

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")          # downloaded samples, uploads, PDF pages

# Several images from https://www.appeconomyinsights.com/
DEMO_IMAGES = {
    "tesla.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbef936e6-3efa-43b3-88d7-7ec620cdb33b_2744x1539.png",
    "netflix.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F23bd84c9-5b62-4526-b467-3088e27e4193_2744x1539.png",
    "nike.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa5cd33ba-ae1a-42a8-a254-d85e690d9870_2741x1541.png",
    "google.png": "https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F395dd3b9-b38e-4d1f-91bc-d37b642ee920_2741x1541.png",
    "accenture.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F08b2227c-7dc8-49f7-b3c5-13cab5443ba6_2741x1541.png",
    "tecent.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0ec8448c-c4d1-4aab-a8e9-2ddebe0c95fd_2741x1541.png",
}


def make_clients(cohere_api_key: str, google_api_key: str):
    """Returns (cohere_client, gemini_client). Imports lazily so the dashboard runs without them."""
    import cohere
    from google import genai
    return cohere.ClientV2(api_key=cohere_api_key), genai.Client(api_key=google_api_key)


# ── Images ────────────────────────────────────────────────────────────────────
def resize_image(pil_image: Image.Image) -> None:
    """Resizes the image in-place if it exceeds MAX_PIXELS."""
    w, h = pil_image.size
    if w * h > MAX_PIXELS:
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        pil_image.thumbnail((int(w * scale), int(h * scale)))


def pil_to_base64(pil_image: Image.Image) -> str:
    img_format = pil_image.format or "PNG"
    resize_image(pil_image)
    with io.BytesIO() as buf:
        pil_image.save(buf, format=img_format)
        return f"data:image/{img_format.lower()};base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_from_image(img_path: str) -> str:
    return pil_to_base64(Image.open(img_path))


# ── Embeddings ────────────────────────────────────────────────────────────────
def embed_image(base64_img: str, co) -> np.ndarray:
    resp = co.embed(model=EMBED_MODEL, input_type="search_document",
                    embedding_types=["float"], images=[base64_img])
    if not (resp.embeddings and resp.embeddings.float_):
        raise RuntimeError("Cohere returned an empty embedding.")
    return np.asarray(resp.embeddings.float_[0])


def embed_query(question: str, co) -> np.ndarray:
    resp = co.embed(model=EMBED_MODEL, input_type="search_query",
                    embedding_types=["float"], texts=[question])
    if not (resp.embeddings and resp.embeddings.float_):
        raise RuntimeError("Cohere returned an empty query embedding.")
    return np.asarray(resp.embeddings.float_[0])


def pdf_to_page_images(pdf_bytes: bytes, pdf_name: str, dpi: int = 150) -> list[str]:
    """Renders each PDF page to a PNG under cache/pdf_pages/<name>/ and returns the paths."""
    import fitz  # PyMuPDF
    out_dir = os.path.join(CACHE_DIR, "pdf_pages", os.path.splitext(pdf_name)[0])
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc.pages(), start=1):
            pix = page.get_pixmap(dpi=dpi)
            path = os.path.join(out_dir, f"page_{i}.png")
            Image.frombytes("RGB", [pix.width, pix.height], pix.samples).save(path, "PNG")
            paths.append(path)
    return paths


def save_upload(data: bytes, name: str) -> str:
    out_dir = os.path.join(CACHE_DIR, "uploads")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def download_demo_images() -> list[str]:
    """Downloads the original demo charts (skips ones already on disk)."""
    out_dir = os.path.join(CACHE_DIR, "demo")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, url in DEMO_IMAGES.items():
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)
        paths.append(path)
    return paths


# ── Retrieval + answer ────────────────────────────────────────────────────────
def search(question: str, co, embeddings: np.ndarray, image_paths: list[str]) -> tuple[str, np.ndarray]:
    """Returns (best image path, similarity score for every image)."""
    if embeddings is None or embeddings.shape[0] != len(image_paths):
        raise ValueError("Embeddings and image list are out of sync — reload the images.")
    query = embed_query(question, co)
    scores = embeddings @ query
    return image_paths[int(np.argmax(scores))], scores


def answer(question: str, img_path: str, gemini) -> str:
    img = Image.open(img_path)
    prompt = [f"""Answer the question based on the following image. Be as elaborate as possible giving extra relevant information.
Don't use markdown formatting in the response.
Please provide enough context for your answer.

Question: {question}""", img]
    return gemini.models.generate_content(model=ANSWER_MODEL, contents=prompt).text
