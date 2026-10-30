"""YOUR FILE. Write decide(): given two institution names, say how they relate.

    "duplicate"  same institution, written differently          ("Science & Technology Council" vs "Science and Technology Council")
    "hierarchy"  one is a part of the other                     ("Cornell University" vs "Cornell University, Roper Center")
    "unrelated"  different institutions                         ("Purdue University" vs "Harvard University")

How to work (error-driven development):
  1. Run:   python evaluate_decider.py            (develops on the TRAIN split; prints scores and the pairs you got wrong)
  2. Read the mistakes. Pick ONE kind of mistake. Change a rule. Run again. Did the score move? Keep it or undo it.
  3. Only when you are happy, run:   python evaluate_decider.py --split test    (do this rarely: it is your final exam)

Level 1 scaffold: the tools are named in the hints, the rules are yours. Look at the cheat sheet (docs/cheatsheet.md) for syntax.
Stuck on syntax?.
"""
import re

from rapidfuzz import fuzz

# ---- settings you can tune (try changing these first) -----------------------------------------------------------
DUPLICATE_FUZZY = 0.90     # fuzzy score at or above this may mean "duplicate"
Replace = {"universität": "university", "dept": "department", "univ": "university", "inst": "institution", "lab": "laboratory", "center": "centre","universitetet": "university","universitat": "university"}

def norm(s: str) -> str:
    """Lowercase, remove punctuation, collapse spaces:  'U.S. Dept.' -> 'u s dept'."""
    NewS = re.sub(r"\s+", " ", re.sub(r"[^\w ]", " ", s.lower())).strip()
    STOP = {"of", "the", "and", "de", "du", "des", "la", "le", "der", "die", "das", "i", "for"}

    result = []
    for word in NewS.split():
        if word not in STOP:
            result.append(Replace.get(word, word)) #It replaces the word with the replacement if it exists, otherwise keeps the original word.
    return " ".join(result)


def decide(name_a: str, name_b: str) -> str:
    na, nb = norm(name_a), norm(name_b)

    # Signals to build rules from (hints, not rules):
    #   fuzz.token_sort_ratio(na, nb) / 100        similarity of the two names, 0 to 1
    #   na in nb   /   nb in na                    does one name sit inside the other?
    #   ("," in name_a) != ("," in name_b)         only one name has a comma: often "Institution, Unit"
    #   len(na.split()), len(nb.split())           how many words in each name

    if(len(na) > len(nb)):
        long_name, short_name = na, nb
    else:
        long_name, short_name = nb, na
    if(len(long_name) != len(short_name)):
        if(long_name.startswith(short_name)):
            Extra = long_name[len(short_name):].split()
            count = len(Extra)
            if("," not in name_a and "," not in name_b and count <= 1):
                return "duplicate"
    
    # TODO 1: if the names are the same after norm(), what are they?
    if na == nb:
        #print("Exact match fired for:", na, nb)
        return "duplicate"
    # TODO 2: if the fuzzy score is very high, what are they?
    if fuzz.token_sort_ratio(na, nb) / 100 >= DUPLICATE_FUZZY:
        #print("Fuzzy score fired for:", na, nb)
        return "duplicate"
    
    # TODO 3: if one name contains the other (and they are not the same), what are they?
    if na in nb or nb in na:
        #print("One name contains the other fired for:", na, nb)
        return "hierarchy"
    
    # TODO 4: everything else.
    # if ("," in name_a) != ("," in name_b):
    #     print("Comma difference fired for:", na, nb)
    #     return "hierarchy"

    return "unrelated"          # starting point: calls everything unrelated. Replace with your rules.
