import xml.etree.ElementTree as ET

NS = {"r3d" : "http://www.re3data.org/schema/4-0"}

def parse_repository(xml_text: str) -> dict:
    root = ET.fromstring(xml_text)
    repo = root.find("r3d:repository",NS)

    repo_id = repo.find("r3d:identifiers/r3d:re3data",NS).text
    name = repo.find("r3d:repositoryName", NS).text
    url = repo.find("r3d:repositoryUrl", NS).text

    doi_elem = repo.find("r3d:identifiers/r3d:doi", NS)         
    doi = doi_elem.text if doi_elem is not None else None

    description_elem = repo.find("r3d:description", NS)
    description = description_elem.text if description_elem is not None else None

    type_elem = repo.find("r3d:type", NS)         
    typology = type_elem.text if type_elem is not None else None

    # --- repeating fields: subjects ---
    subjects = []
    for subj_elem in repo.findall("r3d:subject", NS):
        subjects.append(subj_elem.find("r3d:subjectName", NS).text)

    # --- repeating fields: contacts ---
    contacts = []
    for contact_elem in repo.findall("r3d:repositoryContact", NS):
        info = contact_elem.find("r3d:repositoryContactInformation", NS).text
        ctype_elem = contact_elem.find("r3d:repositoryContactType", NS)
        ctype = ctype_elem.text if ctype_elem is not None else None
        contacts.append({"contact_info": info, "contact_type": ctype})

    return {
        "id": repo_id,
        "name": name,
        "url": url,
        "doi": doi,
        "description": description,
        "typology": typology,
        "subjects": subjects,
        "contacts": contacts,
    }

import requests
if __name__ == "__main__":
    resp = requests.get("https://www.re3data.org/api/v40/repository/r3d100010299")
    print(parse_repository(resp.text))