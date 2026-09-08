import requests
import xml.etree.ElementTree as ET

def get_all_repo_ids() -> list[str]:
    resp =  requests.get("https://www.re3data.org/api/v40/repositories")
    root = ET.fromstring(resp.text)

    ids = []
    for repo_elem in root.findall("repository"):
        id_elem = repo_elem.find("id")
        ids.append(id_elem.text)

    return ids

if __name__ == "__main__":
    ids = get_all_repo_ids()
    print(len(ids))
    print(ids[:3])