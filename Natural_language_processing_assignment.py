# %% 
# NLP Assignment
# 1. IMPORT LIBRARIES
#%matplotlib inline
import os
import re
import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

# 2. LOAD IMDB MOVIE REVIEW DATASET
train_review = load_files(
    r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\aclImdb\train",
    encoding="utf-8"
)

# train_review = load_files('./aclImdb/train/', encoding='utf-8')

x_train, y_train = train_review.data, train_review.target

test_review = load_files(
    r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\aclImdb\test",
    encoding="utf-8"
)

# test_review = load_files('./aclImdb/test/', encoding='utf-8')

x_test, y_test = test_review.data, test_review.target

print("Labels:", train_review.target_names)
print("Sample review:", x_train[0])


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
    from itertools import tee
    # Tokenize
    tokenized = [re.findall(r'\b\w+\b', s.lower()) for s in sentences]

    # Build n-grams
    def ngrams(tokens, n):
        if n == 1:
            return tokens
        else:
            return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    all_ngrams = [ngrams(tokens, ngram) for tokens in tokenized]
    vocab = sorted(list(set([item for sublist in all_ngrams for item in sublist])))

    # Create BoW vectors
    bow_vectors = []
    for ngram_list in all_ngrams:
        vector = [ngram_list.count(word) for word in vocab]
        bow_vectors.append(vector)
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


# 5. TF-IDF USING SKLEARN
X_train_small = x_train[:10]
Y_train_small = y_train[:10]
X_test_small = x_test[:5]
Y_test_small = y_test[:5]

tfidf_vectorizer = TfidfVectorizer(max_features=10, stop_words=stop_words)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_small)
X_test_tfidf = tfidf_vectorizer.transform(X_test_small)

#tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words=stop_words)
#x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
#x_test_tfidf = tfidf_vectorizer.transform(x_test)
#print("\nTF-IDF shape:", x_train_tfidf.shape)


# 6. TRAIN MODEL USING TF-IDF
model = MultinomialNB()
model.fit(x_train_tfidf, y_train)
y_pred = model.predict(x_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("\nTF-IDF Model Accuracy:", accuracy)


# 7. SCRATCH IMPLEMENTATION OF TF-IDF
def tfidf_scratch(sentences):
    tokenized = [re.findall(r'\b\w+\b', s.lower()) for s in sentences]
    vocab = sorted(list(set([item for sublist in tokenized for item in sublist])))
    N = len(sentences)
    
    # Calculate TF
    tf_vectors = []
    for sentence in tokenized:
        tf_vectors.append([sentence.count(word) for word in vocab])
    tf_vectors = np.array(tf_vectors)
    
    # Calculate DF
    df = np.sum(tf_vectors > 0, axis=0)
    
    # Calculate IDF
    idf = np.log(N / (df + 1))
    
    # TF-IDF
    tfidf = tf_vectors * idf
    return pd.DataFrame(tfidf, columns=vocab)

scratch_tfidf = tfidf_scratch([
    "This movie is SOOOO funny!!!",
    "What a movie! I never",
    "best movie ever!!!!! this movie"
])
print("\nScratch TF-IDF:\n", scratch_tfidf)


# 8. WORD2VEC USING GENSIM
# Preprocessing: lowercase + tokenization
sentences = [re.findall(r'\b\w+\b', s.lower()) for s in mini_dataset]

w2v_model = Word2Vec(min_count=1, vector_size=10)
w2v_model.build_vocab(sentences)
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)

# Print vocabulary and vectors
for word in w2v_model.wv.index_to_key:
    print(f"{word} vector:\n{w2v_model.wv[word]}")

# Word similarity example
print("\nMost similar words to 'good':", w2v_model.wv.most_similar('good', topn=3))


# 9. VISUALIZE WORD VECTORS USING TSNE
vocabs = list(w2v_model.wv.index_to_key)
vectors = np.array([w2v_model.wv[word] for word in vocabs])

tsne_model = TSNE(perplexity=5, n_components=2, init="pca", n_iter=200, random_state=23)
vectors_tsne = tsne_model.fit_transform(vectors)

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1])
for i, word in enumerate(vocabs):
    plt.annotate(word, xy=(vectors_tsne[i, 0], vectors_tsne[i, 1]))
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.show()

# %%
