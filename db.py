import psycopg2

def get_connection():
    return psycopg2.connect(
        dbname = "re3data_toolkit",
        host = "localhost",
        user = "postgres",
        password = "r3d",
    )

def insert_repository(cur, data: dict):
    # --- repositories: one row per repo ---
    cur.execute(
        """
        INSERT INTO repositories (id, name, url, doi, description, typology)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            url = EXCLUDED.url,
            doi = EXCLUDED.doi,
            description = EXCLUDED.description,
            typology = EXCLUDED.typology
        """,
        (data["id"], data["name"], data["url"], data["doi"], data["description"], data["typology"]),
    )

    # --- subjects: one row per subject --- #
    for subject in data["subjects"]:
        cur.execute(
            "INSERT INTO subjects (repo_id, subject_name) VALUES (%s, %s)",
            (data["id"],subject),
        )

    # --- contacts: one row per contact --- #
    for contact in data["contacts"]:
        cur.execute(
            "INSERT INTO contacts (repo_id, contact_info,contact_type) VALUES (%s, %s, %s)",

            (data["id"],contact["contact_info"],contact["contact_type"]),

        )

    # --- Institutions: one row per instituion --- #
    for institution in data["institutions"]:
        cur.execute(
            "INSERT INTO institutions (repo_id,institution_name,institution_country,institution_type,institution_url) VALUES(%s, %s, %s, %s, %s)",

            (data["id"],institution["institution_name"],institution["institution_country"],institution["institution_type"],institution["institution_url"]),
        )

# ------------- TESTING ----------------- #

if __name__ == "__main__":
    import requests
    from parse import parse_repository

    conn = get_connection()
    with conn.cursor() as cur:
        for repo_id in ["r3d100010299", "r3d100010468"]:  # WDCC + Zenodo, your two known-good test cases
            resp = requests.get(f"https://www.re3data.org/api/v40/repository/{repo_id}")
            data = parse_repository(resp.text)
            insert_repository(cur, data)
    conn.commit()
    conn.close()
    print("Done — check the database")