# from sentence_transformers import SentenceTransformer

from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from detoxify import Detoxify

# Load model once
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')

embed_model = OllamaEmbeddings(
    model="nomic-embed-text:latest" 
)

def semantic_correctness(expected, answer, threshold=0.80):
    # emb1 = embed_model.encode([expected])
    # emb2 = embed_model.encode([answer])

    v1 = embed_model.embed_query(expected)
    v2 = embed_model.embed_query(answer)

    similarity = cosine_similarity([v1], [v2])[0][0]
    
    is_correct = similarity >= threshold
    return is_correct, similarity


###Build Citation Grounding Check
def citation_grounded(answer, context):
    return answer.strip().lower() in context.lower()

###Build Toxicity Detection
tox_model = Detoxify('original')

def toxicity_score(answer):
    result = tox_model.predict(answer)
    return result["toxicity"]
