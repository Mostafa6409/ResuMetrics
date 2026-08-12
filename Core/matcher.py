from sklearn.metrics.pairwise import cosine_similarity
import joblib

from Core.preprocess import preprocess_Text
from Core.extract import extract


model = joblib.load('../artifacts/BERT_model.joblib')
tfidf = joblib.load('../artifacts/pipeline.joblib')

cv = "cv"
resu = extract("cv")

jd = "job"

def tfidf_pred(cv, jb):
  clean_cv = preprocess_Text(cv)
  clean_jd = preprocess_Text(jb)

  vectors = tfidf.transform([clean_cv, clean_jd])

  raw_tfidf_sim = cosine_similarity(vectors[0], vectors[1])[0][0]

  scaled_tfidf = round(min((raw_tfidf_sim / 0.45) * 100.0, 100.00), 2)

  return scaled_tfidf


def bert_pred(cv, jd):
  embeddings = model.encode([cv, jd])

  raw_bert_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

  scaled_semantic = round(min(raw_bert_sim * 100.0, 100.00), 2)
  return scaled_semantic



def final_score(semantic_score, tfidf_score):

    return (round((0.65 * semantic_score) + (0.35 * tfidf_score), 2))


def final_Result():
 bert_score = bert_pred(cv, jd)
 tfidf_score = tfidf_pred(cv, jd)
 overall_score = final_score(bert_score, tfidf_score)

 return{
        "semantic_score": bert_score,  
        "tfidf_score": tfidf_score,        
        "final_score": overall_score       
    }


