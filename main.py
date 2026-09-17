import time
import logging

from fetch import get_all_repo_ids, get_repository_xml
from parse import parse_repository
from db import get_connection, insert_repository

logging.basicConfig(
    filename = "ingest_failures.log",
    level = logging.INFO,
    format = "%(asctime)s %(message)s"
)

def main():
    ids = get_all_repo_ids()
    print(f"Fetched {len(ids)} repository IDs")

    conn = get_connection()
    success_count = 0
    fail_count = 0

    with conn.cursor() as cur:
        for i, repo_id in enumerate(ids, start = 1):
            try:
                xml_text = get_repository_xml(repo_id)
                data = parse_repository(xml_text)
                insert_repository(cur,data)
                conn.commit()
                success_count += 1
            except Exception as e:
                conn.rollback()
                logging.info(f"{repo_id} FAILED: {e}")
                fail_count += 1
            time.sleep(0.5)

            if(i % 100 == 0):
                print(f"{i}/{len(ids)} processed ({success_count} ok, {fail_count} failed)")
        conn.close()
        print(f"Done. {success_count} succeeded, {fail_count} failed. See ingest_failures.log for details.")

if __name__ == "__main__":
    main()