import time
import logging
from pathlib import Path
from app.db.ncbi_client import ncbi_client
from app.core.ingestion import run_ingestion
from app.utils.config import settings
from app.utils.logging import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger(__name__)

# ------------- CONFIG -------------
QUERY_GROUPS = [
    # GLP-1 drugs split by era to stay under PubMed's 9,999 result cap
    {
        "label": "GLP-1 drugs 2010-2016",
        "keywords": ["GLP-1", "GLP-1 receptor agonist", "exenatide", "liraglutide", "dulaglutide", "Saxenda", "Trulicity"],
        "date_from": "2010/01/01", "date_to": "2016/12/31"
    },
    {
        "label": "GLP-1 drugs 2017-2020",
        "keywords": ["GLP-1", "GLP-1 receptor agonist", "semaglutide", "liraglutide", "exenatide", "dulaglutide", "Ozempic"],
        "date_from": "2017/01/01", "date_to": "2020/12/31"
    },
    {
        "label": "GLP-1 drugs 2021-2023",
        "keywords": ["semaglutide", "tirzepatide", "liraglutide", "GLP-1 receptor agonist", "Ozempic", "Wegovy", "Mounjaro"],
        "date_from": "2021/01/01", "date_to": "2023/12/31"
    },
    {
        "label": "GLP-1 drugs 2024-2026",
        "keywords": ["semaglutide", "tirzepatide", "Ozempic", "Wegovy", "Mounjaro", "GLP-1 receptor agonist"],
        "date_from": "2024/01/01", "date_to": "2026/12/31"
    },
    # Side effects & safety scoped to GLP-1 drugs
    {
        "label": "GLP-1 side effects & safety",
        "keywords": [
            "semaglutide adverse effects", "tirzepatide adverse effects", "liraglutide adverse effects",
            "GLP-1 pancreatitis", "GLP-1 thyroid", "GLP-1 cardiovascular", "GLP-1 nausea",
            "GLP-1 tolerability", "GLP-1 contraindications"
        ],
        "date_from": "2010/01/01", "date_to": "2026/12/31"
    },
    # Nutrition & muscle loss during GLP-1 treatment
    {
        "label": "Nutrition & muscle during GLP-1",
        "keywords": [
            "GLP-1 sarcopenia", "GLP-1 muscle loss", "semaglutide lean body mass",
            "GLP-1 dietary protein", "GLP-1 nutritional deficiency", "GLP-1 nutrition"
        ],
        "date_from": "2010/01/01", "date_to": "2026/12/31"
    },
]

BATCH_SIZE = 50
DELAY_BETWEEN_BATCHES = 0.3  # seconds
MAX_PAPERS = 5000
FOLDER_NAME = "weightloss"
# ----------------------------------


def get_already_saved_pmids() -> set:
    """Return the set of PMIDs already saved to disk so we can skip them."""
    folder = settings.RAW_DATA_DIR / FOLDER_NAME
    if not folder.exists():
        return set()
    return {p.stem for p in folder.glob("*.json")}


def run_ingestion_for_pmids(pmids):
    run_ingestion(keywords=[], limit=0, pmids=pmids, folder_name=FOLDER_NAME)


def main():
    already_saved = get_already_saved_pmids()
    saved_count = len(already_saved)
    logger.info(f"Starting multi-query ingestion. {saved_count} papers already on disk.")

    for group in QUERY_GROUPS:
        if saved_count >= MAX_PAPERS:
            logger.info(f"Reached MAX_PAPERS cap ({MAX_PAPERS}). Stopping.")
            break

        kw_part = " OR ".join(f'"{k}"[tw]' for k in group["keywords"])
        query = f'({kw_part}) AND "free full text"[Filter]'
        query += f' AND ("{group["date_from"]}"[Date - Publication] : "{group["date_to"]}"[Date - Publication])'

        total_hits = ncbi_client.get_total_hits(query)
        logger.info(f"[{group['label']}] {total_hits} hits found.")

        retstart = 0
        while retstart < total_hits and saved_count < MAX_PAPERS:
            pmids = ncbi_client.search_pmids(query, max_results=BATCH_SIZE, retstart=retstart)
            if not pmids:
                logger.warning(f"[{group['label']}] No more PMIDs returned, moving to next group.")
                break

            new_pmids = [p for p in pmids if p not in already_saved]
            skipped = len(pmids) - len(new_pmids)
            if skipped:
                logger.info(f"Skipping {skipped} already-saved PMIDs in this batch.")

            new_pmids = new_pmids[:MAX_PAPERS - saved_count]

            if new_pmids:
                before = len(get_already_saved_pmids())
                run_ingestion_for_pmids(new_pmids)
                after = len(get_already_saved_pmids())
                batch_saved = after - before
                saved_count += batch_saved
                already_saved.update(new_pmids)
                logger.info(f"Progress: {saved_count}/{MAX_PAPERS} total saved.")
            else:
                logger.info(f"Batch {retstart} fully skipped (all already saved).")

            retstart += BATCH_SIZE
            time.sleep(DELAY_BETWEEN_BATCHES)

        logger.info(f"[{group['label']}] Done.")

    logger.info(f"All query groups complete. Total papers saved: {saved_count}")


if __name__ == "__main__":
    main()
