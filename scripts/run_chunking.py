import argparse
import json
import logging
import os
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.chunker import Chunker
from app.utils.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_folder(folder_name: str, max_chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
    """
    Processes all raw JSON files inside a specify input folder name, chunks them,
    and writes the chunked versions to the processed dir.
    """
    input_dir = settings.RAW_DATA_DIR / folder_name
    output_dir = settings.PROCESSED_DATA_DIR / folder_name
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chunker = Chunker(max_chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
    
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {input_dir}")
        return
        
    logger.info(f"Found {len(json_files)} files in {input_dir}. Starting chunking process...")
    
    success_count = 0
    error_count = 0
    
    for file_path in json_files:
        # Skip files that have "_chunked_test" in their name from previous testing
        if "_chunked_test.json" in file_path.name:
            continue
            
        output_filepath = output_dir / file_path.name
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                article_data = json.load(f)
                
            # Process the article into chunks
            docs = chunker.process_article(article_data)
            
            # Serialize LangChain Documents back to JSON dicts
            output_data = {
                "index_a": [
                    {"page_content": c.page_content, "metadata": c.metadata} 
                    for c in docs.get("index_a", [])
                ],
                "index_b": [
                    {"page_content": c.page_content, "metadata": c.metadata} 
                    for c in docs.get("index_b", [])
                ]
            }
            
            # Write to output destination
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
                
            success_count += 1
            if success_count % 10 == 0:
                logger.info(f"Successfully processed {success_count} files so far...")
                
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")
            error_count += 1

    logger.info("====================================")
    logger.info(f"Chunking process complete for '{folder_name}'")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Failed to process: {error_count}")
    logger.info(f"Output saved to: {output_dir}")
    logger.info("====================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk Raw PubMed JSON files into separate indexes.")
    parser.add_argument(
        "--folder", 
        type=str, 
        default="hht", 
        help="The name of the subfolder under data/raw/ to process. Eg: 'hht'."
    )
    parser.add_argument(
        "--max-size", 
        type=int, 
        default=1000, 
        help="Maximum chunk size in characters."
    )
    parser.add_argument(
        "--overlap", 
        type=int, 
        default=200, 
        help="Chunk overlap in characters."
    )
    
    args = parser.parse_args()
    
    process_folder(folder_name=args.folder, max_chunk_size=args.max_size, chunk_overlap=args.overlap)
