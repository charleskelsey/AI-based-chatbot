import cloudscraper
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import os
import time
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
BASE_URL = "https://www.madewithnestle.ca/"
OUTPUT_DIR = "scraped_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "scraped_content_chunks.json")
REQUEST_DELAY = 1  # seconds to wait between requests
MAX_PAGES_TO_SCRAPE = 50 # Limit for testing, remove or increase for full scrape

# --- Helper Functions ---

def is_valid_url(url):
    """Checks if a URL is valid and within the allowed domain."""
    parsed_base = urlparse(BASE_URL)
    parsed_url = urlparse(url)
    return (parsed_url.scheme in ["http", "https"] and
            parsed_url.netloc == parsed_base.netloc and
            not parsed_url.path.endswith(('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.mp4', '.mov', '.avi')) and # Ignore common file types
            not parsed_url.query and # Often skip URLs with query params if they are for filtering/tracking
            "mailto:" not in url and
            "tel:" not in url)


def fetch_page(url):
    """Fetches HTML content of a page."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br', # Server might prefer compressed content
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1', # Do Not Track
            # 'Referer': 'https://www.google.com/' # Sometimes adding a referer helps
        }

        scraper = cloudscraper.create_scraper()
        response = scraper.get(url, timeout=15)
        response.raise_for_status()
        return response.text
    except Exception as e: # cloudscraper might raise different exceptions
        print(f"Error fetching {url} with cloudscraper: {e}")
        return None

def extract_links(html_content, current_url):
    """Extracts all valid, absolute internal links from HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        absolute_link = urljoin(current_url, href) # Handles relative links
        if is_valid_url(absolute_link):
            links.add(absolute_link.split('#')[0]) # Remove fragment identifiers
    return list(links)

def clean_text(text):
    """Basic text cleaning."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
    text = text.strip()
    return text

def extract_meaningful_content(soup, url):
    """
    Extracts meaningful text content from a BeautifulSoup object.
    THIS FUNCTION WILL LIKELY NEED CUSTOMIZATION BASED ON madewithnestle.ca's STRUCTURE.
    """
    title = soup.title.string if soup.title else "No Title"
    title = clean_text(title)

    # Attempt to remove common boilerplate sections
    for selector in ['header', 'footer', 'nav', 'aside', 'script', 'style', '.cookie-banner', '.site-header', '.site-footer', '.navigation', '.sidebar']:
        for element in soup.select(selector):
            element.decompose()

    # --- Content Extraction Logic (NEEDS INSPECTION & ADJUSTMENT) ---
    # General approach: Look for main content containers.
    # Common selectors: 'main', 'article', 'div[role="main"]', 'div.content', 'div.entry-content'
    # You MUST inspect madewithnestle.ca to find appropriate selectors.
    main_content_elements = soup.select('main, article, .main-content, .page-content, .recipe-details, .product-description')
    
    extracted_texts = []

    if not main_content_elements: # Fallback if specific main content selectors fail
        main_content_elements = [soup.body] if soup.body else []

    for content_area in main_content_elements:
        if not content_area: continue

        # General text extraction (headings, paragraphs, list items)
        for tag in content_area.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
            text = clean_text(tag.get_text())
            if text and len(text.split()) > 3 : # Basic filter for very short/empty texts
                extracted_texts.append(text)
        
        # --- Specific extraction for recipes (EXAMPLE - NEEDS VERIFICATION) ---
        # Look for ingredient lists, often in <ul> or <div class="ingredients-list">
        ingredients_section = content_area.select_one('.recipe-ingredients, .ingredients-list, #ingredients')
        if ingredients_section:
            ingredients_text = "Ingredients: "
            for item in ingredients_section.find_all(['li', 'p']):
                ingredients_text += clean_text(item.get_text()) + "; "
            if len(ingredients_text) > len("Ingredients: "):
                 extracted_texts.append(ingredients_text.strip().rstrip(';'))

        # Look for instructions, often in <ol> or <div class="recipe-instructions">
        instructions_section = content_area.select_one('.recipe-instructions, .method-list, #instructions, #method')
        if instructions_section:
            instructions_text = "Instructions: "
            step_count = 1
            for item in instructions_section.find_all(['li', 'p']):
                step_text = clean_text(item.get_text())
                if step_text:
                    instructions_text += f"{step_count}. {step_text} "
                    step_count +=1
            if len(instructions_text) > len("Instructions: "):
                extracted_texts.append(instructions_text.strip())

        # --- Specific extraction for products (EXAMPLE - NEEDS VERIFICATION) ---
        product_desc_el = content_area.select_one('.product-description, .description')
        if product_desc_el:
             extracted_texts.append(f"Product Description: {clean_text(product_desc_el.get_text())}")
        
        nutrition_table = content_area.select_one('.nutrition-facts, .nutrition-table, table[summary="Nutrition Information"]')
        if nutrition_table:
            nutrition_text = "Nutritional Information: "
            for row in nutrition_table.find_all('tr'):
                cells = [clean_text(cell.get_text()) for cell in row.find_all(['td', 'th'])]
                if cells:
                    nutrition_text += ' | '.join(cells) + "; "
            if len(nutrition_text) > len("Nutritional Information: "):
                extracted_texts.append(nutrition_text.strip().rstrip(';'))


    full_page_text = "\n".join(extracted_texts)
    return title, full_page_text


def chunk_page_content(title, page_text, url):
    """Chunks the extracted text using LangChain's text splitter."""
    if not page_text:
        return []

    # Adjust chunk_size and chunk_overlap as needed.
    # chunk_size is in characters. ~1500-2000 chars can be ~400-600 tokens.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,       # Max characters per chunk
        chunk_overlap=200,     # Characters overlap between chunks
        length_function=len,
        add_start_index=True, # Adds start index of chunk in original doc
    )

    # Create documents (LangChain's format) or just split text
    # Adding metadata (url, title) to each chunk is crucial
    docs = text_splitter.create_documents([page_text])
    
    chunked_data = []
    for i, doc in enumerate(docs):
        chunked_data.append({
            "url": url,
            "title": title,
            "chunk_id": i, # Simple incremental ID per page
            "text": doc.page_content, # The actual text chunk
            # "start_index": doc.metadata.get("start_index") # if add_start_index=True
        })
    return chunked_data


# --- Main Scraper Logic ---
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_chunked_data = []
    urls_to_visit = [BASE_URL]
    visited_urls = set()
    pages_scraped_count = 0

    while urls_to_visit and pages_scraped_count < MAX_PAGES_TO_SCRAPE:
        current_url = urls_to_visit.pop(0)
        if current_url in visited_urls:
            continue

        print(f"Scraping ({pages_scraped_count + 1}/{MAX_PAGES_TO_SCRAPE}): {current_url}")
        html_content = fetch_page(current_url)

        if html_content:
            visited_urls.add(current_url)
            pages_scraped_count += 1
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract content and title
            page_title, page_text_content = extract_meaningful_content(soup, current_url)
            
            if page_text_content:
                # Chunk the content
                chunks = chunk_page_content(page_title, page_text_content, current_url)
                if chunks:
                    all_chunked_data.extend(chunks)
                    print(f"  Found {len(page_text_content.split())} words, created {len(chunks)} chunks for {current_url}")
                else:
                    print(f"  No meaningful chunks created for {current_url}")
            else:
                print(f"  No meaningful content extracted from {current_url}")

            # Find new links to visit
            new_links = extract_links(html_content, current_url)
            for link in new_links:
                if link not in visited_urls and link not in urls_to_visit:
                    urls_to_visit.append(link)
            
            time.sleep(REQUEST_DELAY) # Be polite
        else:
            visited_urls.add(current_url) # Add to visited even if fetch failed to avoid retrying constantly

    # Save all chunked data to a single JSON file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunked_data, f, indent=4, ensure_ascii=False)

    print(f"\nScraping complete. Visited {len(visited_urls)} URLs.")
    print(f"Total {len(all_chunked_data)} chunks saved to {OUTPUT_FILE}")
    if pages_scraped_count >= MAX_PAGES_TO_SCRAPE:
        print(f"Stopped early due to MAX_PAGES_TO_SCRAPE limit ({MAX_PAGES_TO_SCRAPE}).")

if __name__ == "__main__":
    main()