from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
from tqdm import tqdm

df = pd.read_csv("src/main/data/product_df_2.csv")  

def combine_product_text(row):
    return " ".join([
        str(row['product_title']) if pd.notnull(row['product_title']) else '',
        str(row['product_description']) if pd.notnull(row['product_description']) else '',
        str(row['product_bullet_point']) if pd.notnull(row['product_bullet_point']) else '',
        str(row['product_brand']) if pd.notnull(row['product_brand']) else '',
        str(row['product_color']) if pd.notnull(row['product_color']) else ''
    ])

df['product_text'] = df.apply(combine_product_text, axis=1)

# 2. Generate embeddings
#model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
model = SentenceTransformer('all-MiniLM-L6-v2')
#model = SentenceTransformer('distilbert-base-uncased')
#model = SentenceTransformer('sentence-transformers/paraphrase-TinyBERT-L6-v2')

product_embeddings = model.encode(df['product_text'].tolist(), show_progress_bar=True)

print("product_embeddings.shape", product_embeddings.shape)

# 3. Build FAISS index on product embeddings
dimension = product_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(product_embeddings).astype('float32'))


# 4. Perform retrieval and compute metrics
hits_at_1, hits_at_5, hits_at_10, mrrs = [], [], [], []

product_ids = df['product_id'].tolist()
grouped = df.groupby('query')


for query, group in tqdm(grouped, desc="Evaluating"):
    query_embedding = model.encode([query])[0].astype('float32').reshape(1, -1)

    D, I = index.search(query_embedding, 10)
    retrieved_product_ids = [product_ids[j] for j in I[0]]

    true_product_ids = group['product_id'].unique().tolist()

    # Check if any of the correct products are in the top-k retrieved results
    hits_at_1.append(int(any(pid in retrieved_product_ids[:1] for pid in true_product_ids)))
    hits_at_5.append(int(any(pid in retrieved_product_ids[:5] for pid in true_product_ids)))
    hits_at_10.append(int(any(pid in retrieved_product_ids[:10] for pid in true_product_ids)))

    # Compute MRR
    rank = next((r+1 for r, pid in enumerate(retrieved_product_ids) if pid in true_product_ids), None)
    mrrs.append(1/rank if rank else 0)



# 5. Print results
print(f"Hits@1: {np.mean(hits_at_1):.3f}")
print(f"Hits@5: {np.mean(hits_at_5):.3f}")
print(f"Hits@10: {np.mean(hits_at_10):.3f}")
print(f"MRR: {np.mean(mrrs):.3f}")