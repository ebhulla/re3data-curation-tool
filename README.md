# re3data Curation Tool

This project pulls repository data from the re3data API and stores it in a
Postgres database. It has four tables: `repositories`, `subjects`,
`contacts`, and `institutions`.

## Project files

- `ingest/fetch.py` — gets data from the re3data API
- `ingest/parse.py` — turns the raw XML into clean fields
- `ingest/db.py` — saves data into the database
- `ingest/main.py` — runs the full pipeline (fetch, parse, save) for all repositories
- `ingest/export.py` — exports the database to an Excel file
- `ingest/schema.sql` — creates the database tables

## Repository layout (reorganized Oct 2026)

Run everything from the repository root, e.g. `python pipeline/judge.py ...`.

| Folder / file | What it holds |
|---|---|
| `decider.py`, `cluster.py`, `evaluate_decider.py` | The rule-based pair decider, the clustering step in progress, and the harness that scores the decider |
| `pipeline/` | Data and labeling pipeline: ROR matching, candidate/mined pairs, the LLM judge, train/val/test split |
| `experiments/` | Baselines, fine-tuning, error analysis, robustness checks, the combined pair model, clustering demo |
| `slurm/` | Negishi job scripts (judge labeling, fine-tuning sweep) |
| `ingest/` | The original re3data -> Postgres ingestion (`fetch`, `parse`, `db`, `main`, `export`, `schema.sql`) |
| `exercises/`, `tests/` | PyTorch exercises (stubs/solutions) and the test suite (`python -m pytest tests -q`) |
| `docs/` | Results notes, learning log, authorship record, Negishi setup, cheat sheet |
| `data/` | Inputs, labels and splits |

## Starting Postgres

Postgres does not start on its own in a new Codespace session. Start it
with:

```bash
sudo service postgresql start
```

Check that it is running:

```bash
sudo service postgresql status
```

You should see `online`.

## Connecting to the database

Use this command to open a connection and run SQL commands directly:

```bash
sudo su - postgres -c "psql -d re3data_toolkit"
```

This puts you inside `psql`, Postgres's command line tool. From here you
can type SQL commands directly. To leave, type `\q` and press Enter.

If you just want to run one command without staying inside `psql`, add
`-c` followed by the command in quotes:

```bash
sudo su - postgres -c "psql -d re3data_toolkit -c 'SELECT count(*) FROM repositories;'"
```

## Viewing the database

**List all tables:**

```bash
sudo su - postgres -c "psql -d re3data_toolkit -c '\dt'"
```

**Count rows in a table:**

```bash
sudo su - postgres -c "psql -d re3data_toolkit -c 'SELECT count(*) FROM repositories;'"
```

**Look at a few rows:**

```bash
sudo su - postgres -c "psql -d re3data_toolkit -c 'SELECT * FROM repositories LIMIT 5;'"
```

**Look at one repository and everything linked to it:**

```bash
sudo su - postgres -c "psql -d re3data_toolkit -c \"SELECT * FROM repositories WHERE id = 'r3d100010299';\""
sudo su - postgres -c "psql -d re3data_toolkit -c \"SELECT * FROM subjects WHERE repo_id = 'r3d100010299';\""
sudo su - postgres -c "psql -d re3data_toolkit -c \"SELECT * FROM contacts WHERE repo_id = 'r3d100010299';\""
sudo su - postgres -c "psql -d re3data_toolkit -c \"SELECT * FROM institutions WHERE repo_id = 'r3d100010299';\""
```

## Setting up the database from scratch

If the tables do not exist yet, or you want to start clean:

```bash
sudo su - postgres -c "psql -d re3data_toolkit -c 'DROP SCHEMA public CASCADE; CREATE SCHEMA public;'"
sudo su - postgres -c "psql -d re3data_toolkit -f ingest/schema.sql"
```

The first command wipes everything. Only run it if you mean to.

## Running the full pipeline

This fetches all repositories from re3data, parses them, and saves them
to the database. It takes about 25 to 30 minutes for all 3,524
repositories.

```bash
source venv/bin/activate
python -u ingest/main.py
```

To run it in the background, so it keeps going even if you close the
terminal:

```bash
nohup python -u ingest/main.py > run_output.log 2>&1 &
```

Check progress with:

```bash
tail -f run_output.log
```

Failed records (if any) are logged to `ingest_failures.log`.

## Exporting to Excel

```bash
python ingest/export.py
```

This creates `r3d_export.xlsx` with one row per repository. Subjects,
contacts, and institutions are joined into one column each, separated by
`; `.

## Common problems

**"Connection refused" when running a database command**
Postgres is not running. Start it with the command at the top of this
file.

**"role does not exist" or a password error**
Make sure you are connecting as the `postgres` user, using
`sudo su - postgres -c "..."` as shown above, not plain `psql`.
