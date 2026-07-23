import json
import logging
import base64
import hashlib
import os
import time
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Try importing PyTorch & OpenCLIP for local vector search
try:
    import torch
    import open_clip
    from PIL import Image
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

logger = logging.getLogger("artwall.intelligence")

# ---------------------------------------------------------------------------
# Cache & Directory Config
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "artwall"
CACHE_FILE = CACHE_DIR / "identifications.json"
CLIP_INDEX_DIR = CACHE_DIR / "clip_index"


class IdentificationCache:
    """Persistent cache of painting identifications, keyed by SHA-256 of the image file."""

    def __init__(self, path: Path = CACHE_FILE):
        self.path = path
        self._data: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
            logger.debug(f"Loaded {len(self._data)} cached identifications.")
        except FileNotFoundError:
            self._data = {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Cache unreadable ({e}); starting fresh.")
            self._data = {}

    def get(self, key: str) -> Optional[Dict[str, str]]:
        entry = self._data.get(key)
        if entry:
            return {k: entry[k] for k in ("title", "artist", "year", "description") if k in entry}
        return None

    def put(self, key: str, identification: Dict[str, str]) -> None:
        entry = dict(identification)
        entry["cached_at"] = int(time.time())
        self._data[key] = entry
        self._save()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as e:
            logger.warning(f"Could not write cache: {e}")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Local CLIP Vector Index
# ---------------------------------------------------------------------------

class LocalCLIPIndex:
    """Lightweight local CLIP vector index for offline artwork identification.
    Uses CPU inference and NumPy dot-product similarity to preserve 100% of VRAM for Ollama."""

    def __init__(self, index_dir: Path = CLIP_INDEX_DIR):
        self.index_dir = index_dir
        self.embeddings_file = self.index_dir / "embeddings.npy"
        self.metadata_file = self.index_dir / "metadata.json"
        
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, str]] = []
        self.model = None
        self.preprocess = None
        
        self._load_index()

    def _load_index(self) -> None:
        if not HAS_CLIP:
            logger.warning("OpenCLIP/PyTorch not installed. Local CLIP search disabled.")
            return
        
        if self.embeddings_file.exists() and self.metadata_file.exists():
            try:
                self.embeddings = np.load(self.embeddings_file)
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded local CLIP index with {len(self.metadata)} artworks.")
            except Exception as e:
                logger.error(f"Failed to load CLIP index files: {e}")
                self.embeddings = None
        else:
            logger.info("No local CLIP index found. Build one using index_artwork() or bulk scripts.")

    def _init_model(self) -> None:
        """Lazy-load the CLIP model onto CPU to keep GPU VRAM completely free for Ollama."""
        if self.model is None and HAS_CLIP:
            logger.debug("Loading CLIP ViT-B-32 onto CPU...")
            # ViT-B-32 is fast and uses minimal RAM (~600MB)
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k', device='cpu'
            )
            self.model.eval()

    def compute_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        if not HAS_CLIP:
            return None
        self._init_model()
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.preprocess(image).unsqueeze(0)
            with torch.no_grad():
                features = self.model.encode_image(tensor)
                features /= features.norm(dim=-1, keepdim=True)  # L2 normalize for cosine similarity
            return features.cpu().numpy()
        except Exception as e:
            logger.warning(f"Failed to compute CLIP embedding for {image_path.name}: {e}")
            return None

    def search(self, image_path: Path, threshold: float = 0.85) -> Optional[Dict[str, Any]]:
        """Searches the local vector index. Threshold >= 0.85 indicates an exact or near-exact image match."""
        if self.embeddings is None or len(self.metadata) == 0:
            return None
        
        query_vec = self.compute_embedding(image_path)
        if query_vec is None:
            return None

        # Fast cosine similarity matrix multiplication in NumPy (~10ms for 100k items)
        similarities = np.dot(self.embeddings, query_vec.T).squeeze()
        
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        
        logger.debug(f"CLIP top match score: {best_score:.3f} ({self.metadata[best_idx].get('title', 'Unknown')})")
        
        if best_score >= threshold:
            result = dict(self.metadata[best_idx])
            result["similarity"] = best_score
            return result
        return None

    def add_to_index(self, image_path: Path, title: str, artist: str, year: str = "") -> bool:
        """Helper to incrementally ingest new gallery images into the local index."""
        vec = self.compute_embedding(image_path)
        if vec is None:
            return False

        meta_entry = {"title": title, "artist": artist, "year": year}
        
        if self.embeddings is None:
            self.embeddings = vec
            self.metadata = [meta_entry]
        else:
            self.embeddings = np.vstack([self.embeddings, vec])
            self.metadata.append(meta_entry)
            
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.embeddings_file, self.embeddings)
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save updated CLIP index: {e}")
            return False


# ---------------------------------------------------------------------------
# Intelligence
# ---------------------------------------------------------------------------

class ArtIntelligence:
    def __init__(self, ollama_url: str, model: str, serpapi_key: str = ""):
        self.url = ollama_url
        self.model = model
        self.serpapi_key = serpapi_key or os.environ.get("SERPAPI_KEY", "")
        self.cache = IdentificationCache()
        self.clip_index = LocalCLIPIndex()

    # -- Temp hosting --

    def _upload_temp(self, image_path: Path) -> Optional[str]:
        try:
            with open(image_path, "rb") as f:
                r = requests.post(
                    "https://tmpfiles.org/api/v1/upload",
                    files={"file": (image_path.name, f, "image/jpeg")},
                    timeout=30,
                )
            r.raise_for_status()
            page_url = r.json()["data"]["url"]
            direct = page_url.replace("tmpfiles.org/", "tmpfiles.org/dl/", 1)
            logger.debug(f"Uploaded to tmpfiles.org: {direct}")
            return direct
        except Exception as e:
            logger.warning(f"tmpfiles.org upload failed: {e}")

        try:
            with open(image_path, "rb") as f:
                r = requests.post(
                    "https://0x0.st",
                    files={"file": (image_path.name, f, "image/jpeg")},
                    data={"expires": "1"},
                    headers={"User-Agent": "artwall/1.0 (personal wallpaper annotator)"},
                    timeout=30,
                )
            r.raise_for_status()
            direct = r.text.strip()
            logger.debug(f"Uploaded to 0x0.st: {direct}")
            return direct
        except Exception as e:
            logger.warning(f"0x0.st upload failed: {e}")

        return None

    # -- SerpApi Google Lens --

    def _search_serpapi(self, image_url: str) -> List[str]:
        logger.info("Querying SerpApi Google Lens...")
        try:
            r = requests.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google_lens",
                    "url": image_url,
                    "hl": "en",
                    "api_key": self.serpapi_key,
                },
                timeout=45,
            )
            r.raise_for_status()
            data = r.json()

            if data.get("error"):
                logger.warning(f"SerpApi error: {data['error']}")
                return []

            titles: List[str] = []
            for kg in data.get("knowledge_graph", []) or []:
                title = kg.get("title", "").strip()
                subtitle = kg.get("subtitle", "").strip()
                if title:
                    titles.append(f"{title} — {subtitle}" if subtitle else title)

            for vm in data.get("visual_matches", []) or []:
                title = vm.get("title", "").strip()
                if title and len(title) > 6:
                    titles.append(title)

            seen = set()
            titles = [t for t in titles if not (t in seen or seen.add(t))][:5]

            if titles:
                logger.debug(f"SerpApi hits: {titles}")
            else:
                logger.warning("SerpApi returned no usable titles.")
            return titles
        except Exception as e:
            logger.warning(f"SerpApi search failed: {e}")
            return []

    def _get_web_context(self, image_path: Path) -> str:
        if not self.serpapi_key:
            logger.info("No SerpApi key configured; skipping web search.")
            return ""
        image_url = self._upload_temp(image_path)
        if not image_url:
            logger.warning("Could not obtain a public URL; skipping web search.")
            return ""
        titles = self._search_serpapi(image_url)
        return " | ".join(titles) if titles else ""

    # -- Ollama Inference Helpers --

    def _call_ollama(self, prompt: str, image_path: Optional[Path] = None) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"num_ctx": 1024, "num_gpu": 99},
        }
        if image_path:
            try:
                with open(image_path, "rb") as f:
                    payload["images"] = [base64.b64encode(f.read()).decode("utf-8")]
            except FileNotFoundError:
                return {}

        try:
            response = requests.post(self.url, json=payload, timeout=90)
            response.raise_for_status()
            return json.loads(response.json().get("response", "{}"))
        except Exception as e:
            logger.error(f"Ollama inference failed: {e}")
            return {}

    # -- Main entry point --

    def identify_via_vlm(self, image_path: Path) -> Dict[str, str]:
        logger.info(f"Analyzing {image_path.name}...")

        # 1. Tier 1: Exact SHA-256 Cache Check
        try:
            cache_key = file_sha256(image_path)
        except FileNotFoundError:
            logger.error(f"File not found: {image_path}")
            return {}

        cached = self.cache.get(cache_key)
        if cached:
            logger.info(f"Cache hit (SHA-256) for {image_path.name}: {cached['title']}")
            return cached

        # 2. Tier 2: Local CLIP Vector Index Check (Offline, Zero Quota Cost)
        clip_match = self.clip_index.search(image_path, threshold=0.85)
        if clip_match:
            logger.info(f"CLIP Index hit ({clip_match['similarity']:.2f}) for {image_path.name}: {clip_match['title']}")
            
            # We already have factual Title/Artist/Year! Ask Ollama only for the visual description.
            prompt = (
                f"You are an art historian. This painting is '{clip_match['title']}' by {clip_match['artist']} ({clip_match.get('year', '')}).\n"
                "Look at the image and write exactly 1-2 sentences summarizing what is depicted visually and its artistic inspiration.\n"
                "Respond ONLY with a JSON object containing keys: 'title', 'artist', 'year', and 'description'.\n"
                f"You MUST set 'title' to \"{clip_match['title']}\", 'artist' to \"{clip_match['artist']}\", and 'year' to \"{clip_match.get('year', '')}\"."
            )
            data = self._call_ollama(prompt, image_path=image_path)
            
            result = {
                "title": clip_match["title"],
                "artist": clip_match["artist"],
                "year": clip_match.get("year", ""),
                "description": str(data.get("description") or "A classic artwork from the gallery index.").strip(),
            }
            self.cache.put(cache_key, result)
            return result

        # 3. Tier 3: Web Context via SerpApi (Only if CLIP misses or low confidence)
        logger.info("Local index miss. Falling back to SerpApi Google Lens...")
        web_context = self._get_web_context(image_path)

        if web_context:
            prompt = (
                f"Here are automated reverse-image search results for this painting from the web:\n"
                f"[{web_context}]\n\n"
                "You are an art historian. Using ONLY the search results above to determine the EXACT title, artist, and year, look at the image. "
                "Respond ONLY with a JSON object containing keys: 'title', 'artist', 'year', and 'description'. "
                "The 'description' field MUST contain exactly 1-2 sentences summarizing what is depicted visually and its artistic inspiration. "
                "Do not guess a different title, artist, or year than what is confirmed in the web search results."
            )
        else:
            # 4. Tier 4: Haiku Fallback Mode
            logger.info("Triggering Haiku Fallback Mode...")
            prompt = (
                "You are a poet. Do not write paragraphs or prose descriptions.\n"
                "Respond ONLY with a JSON object containing keys: 'title', 'artist', 'year', and 'description'.\n"
                "Set 'title' to \"Unknown Painting\", 'artist' to \"Unknown Artist\", and 'year' to \"\".\n"
                "For 'description', write ONLY a 3-line haiku poem about the mood of the painting, separated by '\\n'.\n"
                "Example description: \"Dark waves crash and foam\\nSailboats tilt against the wind\\nOcean claims the sky\""
            )

        data = self._call_ollama(prompt, image_path=image_path)
        result = {
            "title": str(data.get("title") or "Unknown Painting").strip("null ").strip(),
            "artist": str(data.get("artist") or "Unknown Artist").strip("null ").strip(),
            "year": str(data.get("year") or "").strip("null ").strip(),
            "description": str(data.get("description") or "").strip(),
        }

        if web_context and result["title"] != "Unknown Painting":
            self.cache.put(cache_key, result)
            logger.debug(f"Cached identification for {image_path.name}.")
            # Bonus: Automatically feed successful web finds back into your CLIP index!
            self.clip_index.add_to_index(image_path, result["title"], result["artist"], result["year"])

        return result
