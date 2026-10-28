import pandas as pd
from decider import norm
from collections import defaultdict
from itertools import combinations

def cluster(institutions, threshold=50):
    index = defaultdict(list)

    for name in institutions:
        words = norm(name).split()
        for word in words:
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
     
if __name__ == "__main__":
    institutions = pd.read_csv('data/institutions_clean.csv')['institution_name'].dropna().unique()
    judged = pd.concat([pd.read_csv(f"data/pairs_{s}.csv") for s in ["train", "val", "test"]])
    print(len(judged), judged.label.value_counts().to_dict())
    index = cluster(institutions)
    for cutoff in [5,10,20,50,100]:
        pairs = candidate_pairs(index,cutoff)
        print(cutoff,len(pairs))
    
    blocking_recall(pairs,judged,'label')

