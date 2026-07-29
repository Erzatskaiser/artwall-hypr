import os
import csv
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
HEADERS = {"User-Agent": "artwall-ingest/2.0 (personal desktop wallpaper indexer; +https://github.com/)"}

# Use a global session to speed up repeated connections
session = requests.Session()
session.headers.update(HEADERS)


def download_temp_image(url: str) -> Optional[Path]:
    """Streams an image from a URL into a temporary file for VLM embedding."""
    try:
        response = session.get(url, stream=True, timeout=15)
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


def ingest_aic(index: LocalCLIPIndex, target_count: int = 2000):
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
            res = session.get(url, timeout=15).json()
            artworks = res.get("data", [])
            if not artworks:
                break

            for item in artworks:
                if added >= target_count:
                    break

                if item.get("artwork_type_title") != "Painting" or not item.get("image_id"):
                    continue

                title = str(item.get("title") or "Unknown Title").strip()
                artist = str(item.get("artist_title") or "Unknown Artist").strip()
                year = str(item.get("date_display") or "").strip()

                img_url = f"https://www.artic.edu/iiif/2/{item['image_id']}/full/400,/0/default.jpg"

                tmp_path = download_temp_image(img_url)
                if tmp_path and tmp_path.exists():
                    try:
                        success = index.add_to_index(tmp_path, title, artist, year)
                        if success:
                            added += 1
                            logger.info(f"[AIC {added}/{target_count}] Indexed: {title} — {artist}")
                    finally:
                        tmp_path.unlink(missing_ok=True) # Guaranteed cleanup

                time.sleep(0.5)

            page += 1
        except Exception as e:
            logger.error(f"AIC API error on page {page}: {e}")
            break


def ingest_cleveland(index: LocalCLIPIndex, target_count: int = 2000):
    """Scrapes paintings from the Cleveland Museum of Art (CMA) Open Access API."""
    logger.info("=== Starting Ingestion: Cleveland Museum of Art ===")
    added = 0
    skip = 0
    
    while added < target_count:
        url = f"https://openaccess-api.clevelandart.org/api/artworks/?has_image=1&type=Painting&limit=50&skip={skip}"
        try:
            res = session.get(url, timeout=15).json()
            artworks = res.get("data", [])
            if not artworks:
                break
                
            for item in artworks:
                if added >= target_count:
                    break
                    
                images = item.get("images", {})
                web_image = images.get("web", {}).get("url")
                if not web_image:
                    continue
                    
                title = str(item.get("title") or "Unknown Title").strip()
                year = str(item.get("creation_date") or "").strip()
                
                # Safely extract artist from the creators list
                creators = item.get("creators", [])
                artist = str(creators[0].get("description", "Unknown Artist")).strip() if creators else "Unknown Artist"

                tmp_path = download_temp_image(web_image)
                if tmp_path and tmp_path.exists():
                    try:
                        success = index.add_to_index(tmp_path, title, artist, year)
                        if success:
                            added += 1
                            logger.info(f"[CMA {added}/{target_count}] Indexed: {title} — {artist}")
                    finally:
                        tmp_path.unlink(missing_ok=True)
                        
                time.sleep(0.3)
                
            skip += 50
        except Exception as e:
            logger.error(f"CMA API error at skip {skip}: {e}")
            break

def ingest_smk(index: LocalCLIPIndex, target_count: int = 2000):
    """Scrapes paintings from the National Gallery of Denmark (SMK) in English."""
    logger.info("=== Starting Ingestion: National Gallery of Denmark (SMK) ===")
    added = 0
    offset = 0
    
    while added < target_count:
        # Appended &lang=en to force English titles and updated 'maleri' to 'painting'
        url = f"https://api.smk.dk/api/v1/art/search?keys=*&filters=has_image:true AND object_names:painting&offset={offset}&rows=50&lang=en"
        try:
            res = session.get(url, timeout=15).json()
            artworks = res.get("items", [])
            if not artworks:
                break
                
            for item in artworks:
                if added >= target_count:
                    break
                    
                img_url = item.get("image_thumbnail") 
                if not img_url:
                    continue
                    
                # Now these will return the English localized titles
                title = str(item.get("titles", ["Unknown Title"])[0]).strip()
                
                # Extract creator
                production = item.get("production", [])
                artist = str(production[0].get("creator", "Unknown Artist")).strip() if production else "Unknown Artist"
                year = str(item.get("production_dates_notes", [""])[0]).strip()

                tmp_path = download_temp_image(img_url)
                if tmp_path and tmp_path.exists():
                    try:
                        success = index.add_to_index(tmp_path, title, artist, year)
                        if success:
                            added += 1
                            logger.info(f"[SMK {added}/{target_count}] Indexed: {title} — {artist}")
                    finally:
                        tmp_path.unlink(missing_ok=True)
                        
                time.sleep(0.3)
                
            offset += 50
        except Exception as e:
            logger.error(f"SMK API error at offset {offset}: {e}")
            break

def ingest_met(index: LocalCLIPIndex, target_count: int = 2000):
    """Scrapes paintings from The Met with a circuit breaker for rate limits."""
    logger.info("=== Starting Ingestion: The Met Museum ===")
    search_url = "https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=true&medium=Paintings&q=painting"

    try:
        res = session.get(search_url, timeout=15).json()
        object_ids = res.get("objectIDs", [])
        logger.info(f"The Met returned {len(object_ids)} candidate paintings.")
    except Exception as e:
        logger.error(f"Failed to fetch Met search list: {e}")
        return

    added = 0
    consecutive_errors = 0 

    for oid in object_ids:
        if added >= target_count or consecutive_errors >= 10:
            if consecutive_errors >= 10:
                logger.warning("Met API rate limit or connection issues detected. Stopping early.")
            break

        try:
            detail_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}"
            obj = session.get(detail_url, timeout=10).json()
            consecutive_errors = 0

            img_url = obj.get("primaryImageSmall")
            if not img_url:
                continue

            title = str(obj.get("title") or "Unknown Title").strip()
            artist = str(obj.get("artistDisplayName") or "Unknown Artist").strip()
            year = str(obj.get("objectDate") or "").strip()

            tmp_path = download_temp_image(img_url)
            if tmp_path and tmp_path.exists():
                try:
                    success = index.add_to_index(tmp_path, title, artist, year)
                    if success:
                        added += 1
                        logger.info(f"[MET {added}/{target_count}] Indexed: {title} — {artist}")
                finally:
                    tmp_path.unlink(missing_ok=True)

            time.sleep(0.5)
        except Exception as e:
            consecutive_errors += 1
            logger.debug(f"Skipping Met object {oid}: {e}")
            continue


def ingest_wikiart(index: LocalCLIPIndex, target_count: int = 2000):
    """Scrapes the most viewed masterpieces from WikiArt."""
    logger.info("=== Starting Ingestion: WikiArt Most Viewed ===")
    added = 0
    page = 1
    consecutive_errors = 0

    while added < target_count:
        if consecutive_errors >= 5:
            logger.warning("WikiArt API rate limit or connection issues detected. Stopping early.")
            break

        url = f"https://www.wikiart.org/en/api/2/MostViewedPaintings?page={page}"
        try:
            res = session.get(url, timeout=15).json()

            if isinstance(res, dict):
                artworks = res.get("data") or res.get("paintings") or []
            elif isinstance(res, list):
                artworks = res
            else:
                artworks = []

            if not artworks:
                break

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
                    try:
                        success = index.add_to_index(tmp_path, title, artist, year)
                        if success:
                            added += 1
                            logger.info(f"[WikiArt {added}/{target_count}] Indexed: {title} — {artist} ({year})")
                    finally:
                        tmp_path.unlink(missing_ok=True)

                time.sleep(0.3)

            page += 1
            consecutive_errors = 0 

        except Exception as e:
            consecutive_errors += 1
            logger.warning(f"Failed to fetch WikiArt page {page}: {e}")
            time.sleep(2) 
            page += 1

def ingest_nga(index: LocalCLIPIndex, target_count: int = 2000):
    """Scrapes paintings from the National Gallery of Art (NGA) via their Open Data CSVs."""
    logger.info("=== Starting Ingestion: National Gallery of Art (NGA) ===")
    
    # NGA hosts their dataset on GitHub
    base_url = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/"
    
    try:
        # Step 1: Stream and map Object IDs to IIIF Image URLs
        logger.info("Streaming NGA image mappings from GitHub...")
        img_res = session.get(base_url + "published_images.csv", stream=True, timeout=15)
        img_res.raise_for_status()
        
        # Parse the CSV stream line-by-line to save memory
        img_lines = (line.decode('utf-8') for line in img_res.iter_lines())
        img_reader = csv.DictReader(img_lines)
        
        image_map = {}
        for row in img_reader:
            obj_id = row.get("depictstmsobjectid")
            url = row.get("iiifthumburl") or row.get("iiifurl")
            
            if url and obj_id:
                clean_url = url.replace("/full/!200,200/", "/full/!400,400/").replace("/full/full/", "/full/!400,400/")
                image_map[obj_id] = clean_url
                
        logger.info(f"Successfully mapped {len(image_map)} NGA images to memory.")
                
        # Step 2: Stream object metadata and cross-reference
        logger.info("Streaming NGA object metadata...")
        obj_res = session.get(base_url + "objects.csv", stream=True, timeout=15)
        obj_res.raise_for_status()
        
        obj_lines = (line.decode('utf-8') for line in obj_res.iter_lines())
        obj_reader = csv.DictReader(obj_lines)
        
        added = 0
        for row in obj_reader:
            if added >= target_count:
                break
                
            # FIXED: Added .lower() to ensure "Painting" and "painting" both pass the filter
            classification = str(row.get("classification", "")).lower()
            medium = str(row.get("medium", "")).lower()
            
            if classification != "painting" and "painting" not in medium:
                continue
                
            obj_id = row.get("objectid")
            if not obj_id or obj_id not in image_map:
                continue
                
            img_url = image_map[obj_id]
            title = row.get("title", "Unknown Title").strip()
            artist = row.get("attribution", "Unknown Artist").strip() 
            year = row.get("displaydate", "").strip()
            
            tmp_path = download_temp_image(img_url)
            if tmp_path and tmp_path.exists():
                try:
                    success = index.add_to_index(tmp_path, title, artist, year)
                    if success:
                        added += 1
                        logger.info(f"[NGA {added}/{target_count}] Indexed: {title} — {artist}")
                finally:
                    tmp_path.unlink(missing_ok=True)
                    
            time.sleep(0.3)
            
    except Exception as e:
        logger.error(f"Failed to ingest NGA: {e}")

def main():
    logger.info("Initializing Local CLIP Index...")
    index = LocalCLIPIndex()

    # Dictionary of ingest functions to easily toggle or add more in the future
    ingest_sources = [
        (ingest_nga, 3000),
        (ingest_cleveland, 3000),
        (ingest_smk, 3000),
        (ingest_aic, 2000),
        (ingest_wikiart, 2000),
        (ingest_met, 2000)
    ]

    for ingest_func, limit in ingest_sources:
        try:
            ingest_func(index, target_count=limit)
        except Exception as e:
            logger.error(f"Critical failure in {ingest_func.__name__}: {e}")

    # Force a final save to flush the buffer
    index.save_index()

    logger.info("==================================================")
    logger.info(f"Ingestion Complete! Total artworks in database: {len(index.metadata)}")
    logger.info(f"Vector Database saved to: {index.index_dir}")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
