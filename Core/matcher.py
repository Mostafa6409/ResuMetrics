from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from Core.preprocess import preprocess_Text
from Core.extract import extract


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(BASE_DIR, "..", "artifacts", "BERT_model.joblib")
pipeline_PATH = os.path.join(BASE_DIR, "..", "artifacts", "pipeline.joblib")

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(pipeline_PATH)




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


def final_Result(cv,jd):

 cv_text = extract(cv)

 bert_score = bert_pred(cv_text, jd)
 tfidf_score = tfidf_pred(cv_text, jd)
 overall_score = final_score(bert_score, tfidf_score)

 return {
        "semantic_score": round(float(bert_score), 2),
        "tfidf_score": round(float(tfidf_score), 2),
        "final_score": round(float(overall_score), 2)
    }


