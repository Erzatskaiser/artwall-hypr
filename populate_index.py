import os
import time
import logging
import tempfile
import requests
from pathlib import Path
from typing import Optional

# Import the local index from your package
from artwall.intelligence import LocalCLIPIndex

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("artwall.ingest")

# Polite User-Agent required by museum open-data policies
HEADERS = {"User-Agent": "artwall-ingest/1.0 (personal desktop wallpaper indexer)"}


def download_temp_image(url: str) -> Optional[Path]:
    """Streams an image from a URL into a temporary file for VLM embedding."""
    try:
        response = requests.get(url, headers=HEADERS, stream=True, timeout=15)
        if response.status_code != 200:
            return None
            
        # Create a temp file that we will delete after computing the embedding
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            return Path(tmp_file.name)
    except Exception as e:
        logger.debug(f"Failed to download image {url}: {e}")
        return None


def ingest_aic(index: LocalCLIPIndex, target_count: int = 150):
    """Scrapes paintings from the Art Institute of Chicago Open Access API."""
    logger.info("=== Starting Ingestion: Art Institute of Chicago ===")
    added = 0
    page = 1
    
    while added < target_count:
        url = (
            f"https://api.artic.edu/api/v1/artworks?page={page}&limit=50"
            "&fields=id,title,artist_title,date_display,image_id,artwork_type_title"
        )
        try:
            res = requests.get(url, headers=HEADERS, timeout=15).json()
            artworks = res.get("data", [])
            if not artworks:
                break
                
            for item in artworks:
                if added >= target_count:
                    break
                    
                # Ensure it's a painting and has a valid public IIIF image ID
                if item.get("artwork_type_title") != "Painting" or not item.get("image_id"):
                    continue
                
                title = str(item.get("title") or "Unknown Title").strip()
                artist = str(item.get("artist_title") or "Unknown Artist").strip()
                year = str(item.get("date_display") or "").strip()
                
                # IIIF API 2.0 URL: Requesting 400px wide image for fast embedding
                img_url = f"https://www.artic.edu/iiif/2/{item['image_id']}/full/400,/0/default.jpg"
                
                tmp_path = download_temp_image(img_url)
                if tmp_path and tmp_path.exists():
                    success = index.add_to_index(tmp_path, title, artist, year)
                    tmp_path.unlink(missing_ok=True)  # Delete temp file immediately
                    
                    if success:
                        added += 1
                        logger.info(f"[AIC {added}/{target_count}] Indexed: {title} — {artist}")
                
                # AIC guidelines request a ~1 second delay between image downloads to prevent rate-limiting
                time.sleep(0.7)
                
            page += 1
        except Exception as e:
            logger.error(f"AIC API error on page {page}: {e}")
            break


def ingest_met(index: LocalCLIPIndex, target_count: int = 200):
    """Scrapes paintings from The Met with a circuit breaker for rate limits."""
    logger.info("=== Starting Ingestion: The Met Museum ===")
    search_url = "https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=true&medium=Paintings&q=painting"
    
    try:
        res = requests.get(search_url, headers=HEADERS, timeout=15).json()
        object_ids = res.get("objectIDs", [])
        logger.info(f"The Met returned {len(object_ids)} candidate paintings.")
    except Exception as e:
        logger.error(f"Failed to fetch Met search list: {e}")
        return

    added = 0
    consecutive_errors = 0  # Circuit breaker to prevent 30-minute timeout hangs

    for oid in object_ids:
        if added >= target_count or consecutive_errors >= 10:
            if consecutive_errors >= 10:
                logger.warning("Met API rate limit or connection issues detected. Stopping Met ingestion early.")
            break
            
        try:
            detail_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}"
            obj = requests.get(detail_url, headers=HEADERS, timeout=10).json()
            consecutive_errors = 0  # Reset breaker on success
            
            img_url = obj.get("primaryImageSmall")
            if not img_url:
                continue
                
            title = str(obj.get("title") or "Unknown Title").strip()
            artist = str(obj.get("artistDisplayName") or "Unknown Artist").strip()
            year = str(obj.get("objectDate") or "").strip()
            
            tmp_path = download_temp_image(img_url)
            if tmp_path and tmp_path.exists():
                success = index.add_to_index(tmp_path, title, artist, year)
                tmp_path.unlink(missing_ok=True)
                
                if success:
                    added += 1
                    logger.info(f"[MET {added}/{target_count}] Indexed: {title} — {artist}")
            
            time.sleep(0.5)
        except Exception as e:
            consecutive_errors += 1
            logger.debug(f"Skipping Met object {oid}: {e}")
            continue

def ingest_wikiart(index: LocalCLIPIndex, target_count: int = 400):
    """Scrapes the most viewed masterpieces from WikiArt without hardcoding artists."""
    logger.info("=== Starting Ingestion: WikiArt Most Viewed ===")
    
    added = 0
    page = 1
    consecutive_errors = 0
    
    while added < target_count:
        if consecutive_errors >= 5:
            logger.warning("WikiArt API rate limit or connection issues detected. Stopping early.")
            break
            
        # Paginating through the MostViewedPaintings endpoint
        url = f"https://www.wikiart.org/en/api/2/MostViewedPaintings?page={page}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=15).json()
            
            # Safely unwrap the JSON response
            if isinstance(res, dict):
                artworks = res.get("data") or res.get("paintings") or []
            elif isinstance(res, list):
                artworks = res
            else:
                artworks = []

            if not artworks:
                logger.info("No more paintings returned from WikiArt pagination.")
                break

            logger.info(f"WikiArt Page {page} fetched. Processing {len(artworks)} paintings...")
            
            for item in artworks:
                if added >= target_count:
                    break
                
                if not isinstance(item, dict):
                    continue
                    
                img_url = item.get("image")
                if not img_url:
                    continue
                    
                title = str(item.get("title") or "Unknown Title").strip()
                artist = str(item.get("artistName") or "Unknown Artist").strip()
                year = str(item.get("completitionYear") or item.get("yearAsString") or "").strip()
                
                tmp_path = download_temp_image(img_url)
                if tmp_path and tmp_path.exists():
                    success = index.add_to_index(tmp_path, title, artist, year)
                    tmp_path.unlink(missing_ok=True)
                    
                    if success:
                        added += 1
                        logger.info(f"[WikiArt {added}/{target_count}] Indexed: {title} — {artist} ({year})")
                
                time.sleep(0.3)
                
            page += 1
            consecutive_errors = 0  # Reset breaker on a successful page load
            
        except Exception as e:
            consecutive_errors += 1
            logger.warning(f"Failed to fetch WikiArt page {page}: {e}")
            time.sleep(2)  # Give the server a brief pause before retrying the next page
            page += 1
            
    logger.info(f"Finished WikiArt ingestion. Total added: {added}")

def main():
    logger.info("Initializing Local CLIP Index...")
    index = LocalCLIPIndex()
    
    # Ingest Art Institute of Chicago
    ingest_aic(index, target_count=800)
    
    # 2. Ingest The Met
    ingest_met(index, target_count=800)
    
    # 3. Ingest WikiArt
    ingest_wikiart(index, target_count=800)
    
    logger.info("==================================================")
    logger.info(f"Ingestion Complete! Total artworks in database: {len(index.metadata)}")
    logger.info(f"Vector Database saved to: {index.index_dir}")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
