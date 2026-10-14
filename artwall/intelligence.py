import base64
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import requests
from PIL import Image

# Torch and OpenCLIP for offline vector search
try:
    import torch
    import open_clip

    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

from .config import Config

logger = logging.getLogger("artwall.intelligence")


class LocalCLIPIndex:
    """
    Cosine similarity search over a prebuilt matrix of artwork embeddings.

    The model runs on CPU so it does not compete with Ollama for VRAM.
    """

    def __init__(self, index_dir: Path | str = Path.home() / ".cache" / "artwall" / "clip_index"):
        self.index_dir = Path(index_dir).expanduser()
        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.metadata_path = self.index_dir / "metadata.json"

        self.model = None
        self.preprocess = None
        self.tokenizer = None

        self.embeddings = None
        self.metadata = []

        # New vectors are appended here and stacked once at save time. Growing the
        # matrix with vstack on every image copies the whole array each time, which
        # is quadratic over a run of tens of thousands of artworks.
        self._pending = []

        # Buffer to prevent excessive disk writes during large ingestions
        self.unsaved_changes = 0

        # Enforce CPU usage to save VRAM for Ollama inference
        self.device = "cpu"
        self._load_index()

    def _load_index(self):
        """Loads the saved .npy vector matrix and JSON metadata if present."""
        if self.embeddings_path.exists() and self.metadata_path.exists():
            try:
                self.embeddings = np.load(self.embeddings_path)
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(
                    f"Loaded local CLIP index with {len(self.metadata)} artworks."
                )
            except Exception as e:
                logger.error(f"Failed to load local CLIP index: {e}")
        else:
            logger.warning(
                "No local CLIP index found. Run populate_index.py to build one."
            )

    def _init_model(self) -> bool:
        """Lazy-loads the OpenCLIP ViT-B-32 model only when a search or ingest is executed."""
        if not HAS_CLIP:
            logger.warning(
                "OpenCLIP/PyTorch not installed. Local CLIP search disabled."
            )
            return False

        if self.model is None:
            logger.debug(
                f"Loading OpenCLIP ViT-B-32 onto {self.device.upper()}..."
            )
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return True

    def add_to_index(self, image_path: Path, title: str, artist: str, year: str) -> bool:
        """
        Embeds an image and appends it to the index.

        Writes to disk every 50 images rather than on every one, so a long ingest run
        does not spend its time rewriting the same files.
        """
        if not self._init_model():
            return False

        try:
            # 1. Preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # 2. Extract 512-dimensional vector embedding
            with torch.no_grad():
                query_feature = self.model.encode_image(image_tensor)
                query_feature /= query_feature.norm(dim=-1, keepdim=True)
                query_vector = query_feature.cpu().numpy().flatten()

            # 3. Append to memory
            self.metadata.append({
                "title": title,
                "artist": artist,
                "year": year
            })

            self._pending.append(query_vector)

            # Periodic flush so an interrupted run keeps most of its work
            self.unsaved_changes += 1
            if self.unsaved_changes >= 50:
                self.save_index()
                
            return True

        except Exception as e:
            logger.error(f"Error computing embedding for {title}: {e}")
            return False

    def _flush_pending(self):
        """Folds newly computed vectors into the embedding matrix."""
        if not self._pending:
            return
        new = np.asarray(self._pending, dtype=np.float32)
        self.embeddings = new if self.embeddings is None else np.vstack([self.embeddings, new])
        self._pending = []

    def save_index(self):
        """Flushes the in-memory arrays to disk."""
        if self.unsaved_changes == 0:
            return

        self._flush_pending()
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            if self.embeddings is not None:
                np.save(self.embeddings_path, self.embeddings)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=4)
            self.unsaved_changes = 0
        except Exception as e:
            logger.error(f"Failed to save index to disk: {e}")

    def search(self, image_path: Path) -> Tuple[Optional[Dict[str, str]], float]:
        """
        Calculates cosine similarity between the current wallpaper and the local index.
        Always returns a tuple of (best_metadata_dict, best_score).
        """
        self._flush_pending()
        if self.embeddings is None or len(self.metadata) == 0:
            return None, 0.0

        if not self._init_model():
            return None, 0.0

        try:
            # 1. Preprocess image (resize/crop to 224x224, normalize tensor colors)
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # 2. Extract 512-dimensional vector embedding
            with torch.no_grad():
                query_feature = self.model.encode_image(image_tensor)
                query_feature /= query_feature.norm(dim=-1, keepdim=True)
                query_vector = query_feature.cpu().numpy().flatten()

            # 3. Fast Cosine Similarity using NumPy matrix multiplication
            similarities = np.dot(self.embeddings, query_vector)

            # 4. Rank best match
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            best_meta = self.metadata[best_idx]

            logger.debug(
                f"CLIP top match score: {best_score:.3f} ({best_meta['title']})"
            )
            
            return best_meta, best_score

        except Exception as e:
            logger.error(f"Error during local CLIP search: {e}")
            return None, 0.0


class ArtIntelligence:
    """
    Manages the intelligence workflow: Cache -> Vector Search -> VLM Inference.
    Completely offline and zero-API.
    """

    def __init__(self, config: Config):
        self.cfg = config

        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = config.cache_file

        self.known_art = self._load_cache()
        self.clip_index = LocalCLIPIndex(config.index_dir)

    def _load_cache(self) -> dict:
        """Loads previously processed wallpaper hashes from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read cache file: {e}")
        return {}

    def _save_cache(self):
        """Saves metadata associated with calculated SHA-256 hashes to disk."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.known_art, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save metadata cache: {e}")

    def _hash_image(self, image_path: Path) -> str:
        """Generates a SHA-256 hash of the image file."""
        hasher = hashlib.sha256()
        try:
            with open(image_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash image {image_path}: {e}")
            return ""

    def _generate_vlm_description(
        self, image_path: Path, context: str, mode: str = "confident"
    ) -> str:
        """
        Asks the local Ollama model to describe the image.

        The prompts are deliberately restrictive. Without them these models tend to
        narrate object positions ("on the left there is a tree") rather than say
        anything about the painting.
        """
        try:
            with open(image_path, "rb") as f:
                b64_image = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Could not read image for VLM payload: {e}")
            return ""

        if mode == "confident":
            prompt = (
                f"You are a poetic art critic analyzing '{context}'. "
                "Write exactly ONE or TWO concise sentences describing the mood, colors, and brushwork. "
                "CRITICAL INSTRUCTIONS: Do NOT use phrases like 'The image displays', 'This painting is', or 'In the foreground'. "
                "Do NOT give spatial coordinates. Do NOT list objects. Jump straight into the stylistic description."
            )
        elif mode == "doubtful":
            logger.info("Triggering Doubtful/Fuzzy Mode...")
            prompt = (
                f"You are a poetic art critic. This piece strongly resembles the style of '{context}'. "
                "Write exactly ONE or TWO concise sentences describing the mood, lighting, and technique you see. "
                "CRITICAL INSTRUCTIONS: Do NOT use phrases like 'The image displays', 'There is a', or 'On the left'. "
                "Do NOT list objects. Describe the artistic impression directly and beautifully. Keep it under 30 words."
            )
        else:
            logger.info("Triggering Haiku Fallback Mode...")
            prompt = (
                "Write a beautiful, evocative 3-line haiku about the mood, colors, and scenery "
                "shown in this image. Do not include any titles or artist names. "
                "Just output the 3 lines of the haiku."
            )

        payload = {
            "model": self.cfg.vlm_model,
            "prompt": prompt,
            "stream": False,
            "images": [b64_image],
        }

        try:
            logger.debug(f"Sending request to Ollama ({self.cfg.vlm_model})...")
            start_time = time.time()
            res = requests.post(self.cfg.ollama_url, json=payload, timeout=self.cfg.ollama_timeout)
            res.raise_for_status()

            description = res.json().get("response", "").strip()
            logger.debug(
                f"Ollama inference completed in {time.time() - start_time:.1f}s"
            )
            return description

        except requests.exceptions.RequestException as e:
            logger.error(
                f"Ollama API request failed. Ensure the service is running: {e}"
            )
            return "Local AI service unavailable."

    def process_wallpaper(self, image_path: Path) -> Dict[str, str]:
        """
        Main execution pipeline:
        1. Cache Lookup (SHA-256)
        2. Local OpenCLIP Search
        3. Ollama VLM Inference (Confident, Doubtful, or Haiku configurable)
        """
        logger.info(f"Analyzing {image_path.name}...")
        img_hash = self._hash_image(image_path)

        # Already seen this exact file
        if img_hash and img_hash in self.known_art:
            logger.info(
                "Identical image found in local cache. Skipping AI analysis."
            )
            return self.known_art[img_hash]

        meta = {
            "title": "Unknown Art",
            "artist": "Local Collection",
            "year": "",
            "description": "",
        }

        # Look for the closest artwork in the index
        clip_match, score = self.clip_index.search(image_path)

        if not clip_match:
             meta["description"] = "No local database found. Run populate_index.py."
             return meta

        context_str = f"{clip_match['title']} by {clip_match['artist']} ({clip_match.get('year', '')})"

        # Decide how to report the match. Thresholds live in the config.
        if score >= self.cfg.confident_score:
            # Close enough to name the artwork outright
            meta["title"] = clip_match["title"]
            meta["artist"] = clip_match["artist"]
            meta["year"] = clip_match.get("year", "")
            meta["description"] = self._generate_vlm_description(
                image_path, context=context_str, mode="confident"
            )
            
        elif score >= self.cfg.fuzzy_score or self.cfg.fallback_mode == "reminiscent":
            # A resemblance, or a weak match the config still wants reported as one
            meta["title"] = f"Reminiscent of: {clip_match['title']}"
            meta["artist"] = clip_match["artist"]
            meta["year"] = clip_match.get("year", "")
            meta["description"] = self._generate_vlm_description(
                image_path, context=context_str, mode="doubtful"
            )
            
        else:
            # Nothing close in the index
            logger.info("Local vector miss. Running local VLM haiku generator.")
            meta["description"] = self._generate_vlm_description(
                image_path, context="", mode="unknown"
            )

        # Update cache
        if img_hash:
            self.known_art[img_hash] = meta
            self._save_cache()

        return meta
