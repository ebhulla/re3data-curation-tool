import pandas as pd
from decider import norm
from collections import defaultdict,Counter
from decider import decide
from itertools import combinations


CUTOFF = 150

def cluster(institutions, threshold=50):
    index = defaultdict(list)

    for name in institutions:
        words = norm(name).split()
        for word in set(words):
                index[word].append(name)
    #print(len(institutions))                    # how many names did we load?
    #print(len(index))                           # how many distinct words are in the dictionary?
    #print(list(index.items())[:2])              # the first two entries



    #print(len(index["wellcome"]))
    #print(len(index["university"]))
    return index

def candidate_pairs(index,cutoff):
     pairs = set()
     for word, names in index.items():
        if(len(names) > cutoff):
                continue
        else:
            for name1, name2 in combinations(names, 2):
                pairs.add((name1, name2))
     return pairs

def blocking_recall(pairs,judged,label):
    dups = judged[judged.label == "duplicate"]
    keys = [tuple(sorted((a,b))) for a,b in zip(dups.name_a,dups.name_b)]
    return keys

parent={}
def find(a):
     if(a not in parent):
         parent[a] = a
     while(parent[a] != a):
            a = parent[a]
     return a
def union(a,b):
    root_a = find(a)
    root_b = find(b)
    if root_a != root_b:
        parent[root_b] = root_a

if __name__ == "__main__":
    institutions = pd.read_csv('data/institutions_clean.csv')['institution_name'].dropna().unique()
    judged = pd.concat([pd.read_csv(f"data/pairs_{s}.csv") for s in ["train", "val", "test"]])
    #print(len(judged), judged.label.value_counts().to_dict())
    
    index = cluster(institutions)
    # for cutoff in [50, 100, 150, 200, 300, 400, 500]: #Inititally tested until 1000, stays flat after 500
    #     pairs = candidate_pairs(index, cutoff)
    #     keys = blocking_recall(pairs,judged,'label')
    #     found = sum(1 for pair in pairs if tuple(sorted(pair)) in keys)
    #     #print("cutoff:", cutoff, "candidate pairs:", len(pairs)," found by blocking:", found, " recall:", found / len(keys)) 
    #     #print("duplicates judged:", len(keys))
        

    # missed = [k for k in keys if k not in pairs]
    # for a,b in missed[:15]:
    #      print(a,"|",b)
    pairs = candidate_pairs(index, CUTOFF)
    answers = Counter(decide(a,b) for a,b in pairs)
    print(answers)
    dup_edges = [(a, b) for a, b in pairs if decide(a, b) == "duplicate"]
    hier_edges = [(a, b) for a, b in pairs if decide(a, b) == "hierarchy"]
    # print(len(dup_edges), len(hier_edges))        # expect 925 and 3287

    # for a, b in dup_edges[:5]:
    #     print("DUP |", a, "|", b)
    # for a, b in hier_edges[:5]:
    #     print("HIER|", a, "|", b)
    for a, b in dup_edges:
        union(a, b)                      # each duplicate edge joins two groups

    clusters = defaultdict(list)         # defaultdict is already imported in your file
    for name in institutions:            # every distinct name
        clusters[find(name)].append(name)

    sizes = sorted((len(members) for members in clusters.values()), reverse=True)
    print("names:", len(institutions), " clusters:", len(clusters))
    print("five largest:", sizes[:5])
    biggest = max(clusters.values(), key=len)
    for name in biggest:
        print(name)
    