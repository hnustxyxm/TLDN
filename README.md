# TLDN
Here is the source code of "TLDN: Reinforcing Web Service Classification via Combining Various Unstructured and Structured Features"

## Requirement:
* python: 3.6
* gensim: 3.6.0
* numpy: 1.19.0
* pandas: 0.23.4
* lda: 1.1.0
* argparse: 1.4.0
* networkx: 2.2
* torch: 1.5.0
* matplotlib: 3.0.0

## Training steps : (Run the following program in turn )

### ./LDA and Data Preprocessing file:
* Mashuppre_LDA_20_cate.py
* APIpre_LDA_20_cate.py

### ./TF-IDF file:
* TF-IDF_mashup_API_20_cate.py

### ./Doc2vec and Data Preprocessing file:
* Doc2API_20_cate.py
* Doc2mashup_20_cate.py

### ./Node2vec file:
* Node2_mashup_20_cate_generation.py
* Node2_API_20_cate_generation.py
* graph_data to edgelist_data.py
* node2vec_main1.py
* emb_data to fina_data.py

### ./train_test_data_generationfile:
* mashup_train_test_data_generation.py
* API_train_test_data_generation.py

### TFIDF_Doc2vec_LDA_Node2vec file:
* mashup_best_param_Classification.py
* API_best_param_Classification.py
* mashup_compare_Classification.py
* API_compare_Classification.py
