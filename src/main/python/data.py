"""
This script processes two datasets: shopping queries and product details. 

It performs the following steps:
1. Loads `shopping_queries_dataset_examples.parquet` and `shopping_queries_dataset_products.parquet`.
2. Merges them on `product_locale` and `product_id`.
3. Filters the data ESCI label "E" and product_locale us
4. Keeps relevant columns: `query`, `query_id`, `product_id`, `product_locale`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color`, and `esci_label`.
5. Saves the filtered dataset as a CSV file.

Output:
- A CSV file containing the filtered data.
"""


import pandas as pd

df_examples = pd.read_parquet('src/main/data/shopping_queries_dataset_examples.parquet')
df_products = pd.read_parquet('src/main/data/shopping_queries_dataset_products.parquet')

# Merge datasets
df_example_products = pd.merge(
    df_examples,
    df_products,
    how="left",
    left_on=["product_locale", "product_id"],
    right_on=["product_locale", "product_id"],
)

# Get the shape of df_example_products
print("Shape of df_example_products:", df_example_products.shape)


# Filter for queries (case-insensitive) and ESCI label "E"
df_filtered = df_example_products[
   (df_example_products["esci_label"] == "E") & (df_example_products["product_locale"] == "us")
]


print("Shape of filtered df_example_products:", df_filtered.shape)


# Step 1: Select 50 unique queries randomly
# unique_queries = df_filtered['query'].dropna().unique()
# selected_queries = pd.Series(unique_queries).sample(50, random_state=42)


# filtered_df = df_filtered[df_filtered['query'].isin(selected_queries)]

# final_sample = filtered_df.sample(500, random_state=42)
# final_sample = final_sample.reset_index(drop=True)


# print("final sample", final_sample.shape)

# grouped = final_sample['query'].value_counts()

# print("samples per query", grouped)

# # Filter only product columns
# product_columns = [col for col in final_sample.columns if col.startswith('product_')]
# product_df = final_sample[product_columns + ['query']]

# print(product_df.shape)
# print(product_df.columns)

# #df_final = df_filtered.head(100)

# output_path = "src/main/data/product_df.csv"
# product_df.to_csv(output_path, index=False)


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Select 50 unique queries randomly (as before)
unique_queries = df_filtered['query'].dropna().unique()

# Step 2: Vectorize queries for clustering
vectorizer = TfidfVectorizer(stop_words='english')
query_vectors = vectorizer.fit_transform(unique_queries)

# Step 3: Apply clustering (e.g., KMeans) to group similar queries together
num_clusters = 10  # You can change this based on your needs
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(query_vectors)

# Step 4: Assign each query to a cluster
query_clusters = pd.Series(kmeans.labels_, index=unique_queries)

# Step 5: Pick a query from each cluster to ensure diversity
sampled_queries = []

# Select 5 queries from each cluster
for cluster_id in range(num_clusters):
    cluster_queries = query_clusters[query_clusters == cluster_id].index
    sampled_queries.extend(np.random.choice(cluster_queries, size=5, replace=False))



print("sampled_queries", len(sampled_queries))
# Step 6: Filter the dataframe to get all rows containing the selected queries
filtered_df = df_filtered[df_filtered['query'].isin(sampled_queries)]

# Step 7: Further sample 500 rows (optional) for final sample
# final_sample = filtered_df.sample(500, random_state=42)
# final_sample = final_sample.reset_index(drop=True)


from sklearn.model_selection import train_test_split

# Ensure 'query' column is treated as a stratification label
final_sample, _ = train_test_split(
    filtered_df,
    train_size=500,
    stratify=filtered_df['query'],
    random_state=42
)
final_sample = final_sample.reset_index(drop=True)


# Step 8: Verify the results
print("Final sample size:", final_sample.shape)
grouped = final_sample['query'].value_counts()
print("Samples per query:", grouped)

# Filter only product columns
product_columns = [col for col in final_sample.columns if col.startswith('product_')]
product_df = final_sample[product_columns + ['query']]

# Check out the result
print(product_df.head())


# #df_final = df_filtered.head(100)

output_path = "src/main/data/product_df_2.csv"
product_df.to_csv(output_path, index=False)

