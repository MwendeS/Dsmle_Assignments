# %% 
# NLP Assignment - Optimized for speed
# 1. IMPORT LIBRARIES
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter

# Robust NLTK stopwords load
try:
    stop_words = stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = stopwords.words('english')

# Function to load a limited number of files per class
def load_limited_files(path, n_files_total=500):
    texts, labels = [], []
    class_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    n_per_class = max(1, n_files_total // len(class_dirs))
    for class_idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(path, class_name)
        files = os.listdir(class_path)[:n_per_class]
        for file in files:
            with open(os.path.join(class_path, file), encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())
                labels.append(class_idx)
    return texts, labels

# 2. LOAD IMDB DATASET (limited)
train_path = r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\aclImdb\train"
test_path  = r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\aclImdb\test"

x_train, y_train = load_limited_files(train_path, n_files_total=500)
x_test, y_test = load_limited_files(test_path, n_files_total=100)

print("Loaded", len(x_train), "train reviews and", len(x_test), "test reviews.")

# 3. BAG OF WORDS (BoW) EXAMPLE
mini_dataset = [
    "This movie is very good.",
    "This film is a good",
    "Very bad. Very, very bad."
]

vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
bow = vectorizer.fit_transform(mini_dataset).toarray()
df_bow = pd.DataFrame(bow, columns=vectorizer.get_feature_names_out())
print("\nBoW Example:\n", df_bow)

# 3.1 2-GRAM Example
vectorizer_bigram = CountVectorizer(ngram_range=(2,2), token_pattern=r'(?u)\b\w+\b')
bow_bigram = vectorizer_bigram.fit_transform(mini_dataset).toarray()
df_bow_bigram = pd.DataFrame(bow_bigram, columns=vectorizer_bigram.get_feature_names_out())
print("\n2-Gram BoW Example:\n", df_bow_bigram)

# 4. SCRATCH IMPLEMENTATION OF BoW
def bow_scratch(sentences, ngram=1):
    tokenized = [re.findall(r'\b\w+\b', s.lower()) for s in sentences]

    def ngrams(tokens, n):
        if n == 1:
            return tokens
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    all_ngrams = [ngrams(tokens, ngram) for tokens in tokenized]
    vocab = sorted(list(set([item for sublist in all_ngrams for item in sublist])))

    bow_vectors = []
    for ngram_list in all_ngrams:
        c = Counter(ngram_list)
        bow_vectors.append([c[word] for word in vocab])
    return pd.DataFrame(bow_vectors, columns=vocab)

scratch_bow_1gram = bow_scratch([
    "This movie is SOOOO funny!!!",
    "What a movie! I never",
    "best movie ever!!!!! this movie"
], ngram=1)
print("\nScratch BoW 1-Gram:\n", scratch_bow_1gram)

scratch_bow_2gram = bow_scratch([
    "This movie is SOOOO funny!!!",
    "What a movie! I never",
    "best movie ever!!!!! this movie"
], ngram=2)
print("\nScratch BoW 2-Gram:\n", scratch_bow_2gram)

# 5. TF-IDF USING SKLEARN (limited subset)
X_train_small = x_train[:200]
Y_train_small = y_train[:200]
X_test_small = x_test[:50]
Y_test_small = y_test[:50]

tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words=stop_words)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_small)
X_test_tfidf = tfidf_vectorizer.transform(X_test_small)

# 6. TRAIN MODEL USING TF-IDF
model = MultinomialNB()
model.fit(X_train_tfidf, Y_train_small)
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(Y_test_small, y_pred)
print("\nTF-IDF Model Accuracy:", accuracy)

# 7. SCRATCH IMPLEMENTATION OF TF-IDF
def tfidf_scratch(sentences):
    tokenized = [re.findall(r'\b\w+\b', s.lower()) for s in sentences]
    vocab = sorted(list(set([item for sublist in tokenized for item in sublist])))
    N = len(sentences)
    
    tf_vectors = []
    for sentence in tokenized:
        tf_vectors.append([sentence.count(word) for word in vocab])
    tf_vectors = np.array(tf_vectors)
    
    df = np.sum(tf_vectors > 0, axis=0)
    idf = np.log(N / (df + 1)) + 1.0
    
    tfidf = tf_vectors * idf
    return pd.DataFrame(tfidf, columns=vocab)

scratch_tfidf = tfidf_scratch([
    "This movie is SOOOO funny!!!",
    "What a movie! I never",
    "best movie ever!!!!! this movie"
])
print("\nScratch TF-IDF:\n", scratch_tfidf)

# 8. WORD2VEC USING GENSIM (faster)
sentences = [re.findall(r'\b\w+\b', s.lower()) for s in mini_dataset]
w2v_model = Word2Vec(sentences=sentences, vector_size=10, min_count=1, workers=min(4, os.cpu_count()), epochs=10)

for word in w2v_model.wv.index_to_key:
    print(f"{word} vector:\n{w2v_model.wv[word]}")

print("\nMost similar words to 'good':", w2v_model.wv.most_similar('good', topn=3))

# 9. VISUALIZE WORD VECTORS USING TSNE (fast)
vocabs = list(w2v_model.wv.index_to_key)
vectors = np.array([w2v_model.wv[word] for word in vocabs])
tsne_model = TSNE(perplexity=3, n_components=2, init="pca", max_iter=250, random_state=23)
vectors_tsne = tsne_model.fit_transform(vectors)

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1])
for i, word in enumerate(vocabs):
    plt.annotate(word, xy=(vectors_tsne[i, 0], vectors_tsne[i, 1]))
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.show()
# %%
