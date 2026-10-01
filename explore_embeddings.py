from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

names = ['Purdue University', 'Indiana University', 'University of Michigan','University of Illinois','University of Wisconsin','University of Minnesota', 'University of Iowa','University of California Santa Barbara']

vectors = model.encode(names)
score = {}
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for pair in zip(names, vectors):
     #score[pair[0]] = cosine_similarity(pair[1], vectors[0]) 
     #pair[0] is the name and pair[1] is the vector
     for i in range(len(vectors)):
            score[(pair[0], names[i])] = cosine_similarity(pair[1], vectors[i])
            print(f"Similarity between {pair[0]} and {names[i]}: {cosine_similarity(pair[1], vectors[i])}")

print(vectors.shape)    
print(score)