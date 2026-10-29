import pandas as pd
from db import get_connection

QUERY = """
SELECT
    r.id, r.name, r.url, r.doi, r.description, r.typology,
    string_agg(DISTINCT s.subject_name, '; ') AS subjects,
    string_agg(DISTINCT c.contact_info, '; ') AS contacts,
    string_agg(DISTINCT i.institution_name, '; ') AS institutions
FROM repositories r
LEFT JOIN subjects s ON s.repo_id = r.id
LEFT JOIN contacts c ON c.repo_id = r.id
LEFT JOIN institutions i ON i.repo_id = r.id
GROUP BY r.id, r.name, r.url, r.doi, r.description, r.typology
"""

conn = get_connection()
df = pd.read_sql(QUERY, conn)
df.to_excel("r3d_export.xlsx", index=False)
conn.close()
print(f"Exported {len(df)} rows")