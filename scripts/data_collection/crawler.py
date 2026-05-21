import os
import requests
import config
from typing import List
import time

class WikimediaCrawler:
    """
    A simple crawler to fetch images of Vietnamese landmarks from Wikimedia Commons.
    """
    def __init__(self, download_dir: str = config.DATASET_DIR):
        self.base_url = "https://commons.wikimedia.org/w/api.php"
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        self.headers = {
            # Standard format: <client name>/<version> (<contact information>) <library/framework name>/<version>
            "User-Agent": f"DinoCBIRBot/1.0 (contact: maithetranh@gmail.com) requests/{requests.__version__}",
            "Api-User-Agent": "DinoCBIRBot/1.0 (contact: maithetranh@gmail.com)"
        }

    def search_images(self, query: str, limit: int = 50) -> List[dict]:
        """
        Searches for images on Wikimedia Commons based on a query.
        """
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": 6,  # File namespace
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url"
        }

        response = requests.get(self.base_url, params=params, headers=self.headers)
        response.raise_for_status()
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        images = []
        for page_id, page_data in pages.items():
            if "imageinfo" in page_data:
                img_url = page_data["imageinfo"][0]["url"]
                title = page_data["title"].replace("File:", "").replace(" ", "_")
                images.append({"url": img_url, "title": title})
        
        return images

    def download_images(self, images: List[dict]):
        """
        Downloads a list of images to the local directory.
        """
        print(f"Starting download of {len(images)} images to {self.download_dir}...")
        for i, img in enumerate(images):
            try:
                # Clean filename
                filename = "".join([c for c in img["title"] if c.isalnum() or c in "._-"]).strip()
                save_path = os.path.join(self.download_dir, filename)
                
                if os.path.exists(save_path):
                    continue

                response = requests.get(img["url"], headers=self.headers, stream=True)
                response.raise_for_status()
                
                # Verify content type to ensure it's an image and not an error page
                content_type = response.headers.get("Content-Type", "")
                if "image" not in content_type:
                    print(f"Skipping {img['url']}: Not an image (Content-Type: {content_type})")
                    continue

                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"[{i+1}/{len(images)}] Downloaded: {filename}")
                # Polite crawling
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed to download {img['url']}: {e}")

if __name__ == "__main__":
    import json
    
    # Load landmarks from kb.json
    try:
        with open(config.KB_JSON_PATH, "r", encoding="utf-8") as f:
            kb_nodes = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {config.KB_JSON_PATH}")
        kb_nodes = []

    print(f"Loaded {len(kb_nodes)} nodes from Knowledge Base.")
    
    # Filter only objects (landmarks)
    landmarks = [node for node in kb_nodes if node.get("type") == "object"]
    
    if not landmarks:
        print("No landmarks found in kb.json.")
        exit()

    for node in landmarks:
        node_id = node.get("kb_id") or node.get("id")
        name_en = node.get("name")
        
        if not node_id or not name_en:
            continue
            
        print(f"\n[{node_id}] Searching for images of: {name_en}")
        
        # Create a specific directory for this landmark
        target_dir = os.path.join(config.DATASET_DIR, node_id)
        crawler = WikimediaCrawler(download_dir=target_dir)
        
        # Search using the English name. You could append ' Hanoi' to improve accuracy.
        query = f"{name_en} Hanoi"
        images = crawler.search_images(query, limit=10) # 10 images should be plenty for reference
        
        if images:
            crawler.download_images(images)
        else:
            print(f"No images found for {query}")
    
    print("\nCrawling process finished.")
