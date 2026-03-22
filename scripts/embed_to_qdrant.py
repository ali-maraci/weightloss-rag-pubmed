"""
Reads processed JSON files from data/processed/weightloss/ and embeds
them directly into Qdrant Cloud using AuraEmbedder.
Skips any PMIDs already present in Qdrant to allow safe re-runs.
"""
import json
import logging
import argparse
from pathlib import Path

from app.core.embedder import AuraEmbedder
from app.utils.config import settings
from app.utils.logging import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def main(folder_name: str):
    input_dir = settings.PROCESSED_DATA_DIR / folder_name

    if not input_dir.exists():
        logger.error(f"Processed data directory not found: {input_dir}")
        return

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        logger.error(f"No processed JSON files found in {input_dir}")
        return

    logger.info(f"Found {len(json_files)} processed files. Starting embedding...")
    embedder = AuraEmbedder()

    success, skipped, failed = 0, 0, 0

    for i, file_path in enumerate(json_files, 1):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                article_data = json.load(f)

            result = embedder.ingest_article(article_data)

            if result:
                # AuraEmbedder returns True for both newly embedded and skipped (already exists)
                success += 1
            else:
                failed += 1

            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(json_files)} files processed "
                            f"({success} ok, {failed} failed)")

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            failed += 1

    logger.info("==========================================")
    logger.info(f"Embedding complete for folder '{folder_name}'")
    logger.info(f"Total processed: {len(json_files)}")
    logger.info(f"Success: {success} | Failed: {failed}")
    logger.info("==========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed processed PubMed chunks into Qdrant.")
    parser.add_argument(
        "--folder",
        type=str,
        default="weightloss",
        help="Subfolder under data/processed/ to embed. Default: weightloss"
    )
    args = parser.parse_args()
    main(folder_name=args.folder)
