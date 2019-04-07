import pyLDAvis
import pyLDAvis.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from glob import glob
from textblob import TextBlob as tb
import pandas as pd

###########################
####LEMMATIZE FIRST!!!!####
###########################

path = "/Users/karanjani/Desktop/csvwithCleanedVecs/*.csv"
files = glob(path)
docs_raw = []
utterance_temp = []

# for doc in files: #Meant to read in entire txt files
# 	with open (doc, 'r') as f:
# 		data = f.read()                      
# 		docs_raw.append(data)

for file in files: #Topic modeling that reads in utterances
	df = pd.read_csv(file)
	utterance_temp.append(df['stringList'].tolist())
	
utterance_raw = [item for sublist in utterance_temp for item in sublist]

tf_vectorizer = CountVectorizer(strip_accents = 'unicode',stop_words = 'english',lowercase=True,token_pattern = r'\b[a-zA-Z]{3,}\b',max_df = 0.2,min_df = 0)
dtm_tf = tf_vectorizer.fit_transform(utterance_raw)

tfidf_vectorizer = TfidfVectorizer(**tf_vectorizer.get_params())
dtm_tfidf = tfidf_vectorizer.fit_transform(utterance_raw)

# for TF DTM
lda_tf = LatentDirichletAllocation(n_topics=30, random_state=0)
lda_tf.fit(dtm_tf)
# for TFIDF DTM
lda_tfidf = LatentDirichletAllocation(n_topics=30, random_state=0)
lda_tfidf.fit(dtm_tfidf)

nmf_tf = NMF(n_components=80, random_state=1,
          alpha=.1, l1_ratio=.5).fit(dtm_tf)

# nmf_tfidf = NMF(n_components=10, random_state=1,
#           beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
#           l1_ratio=.5).fit(dtm_tfidf)

vis1 = pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer)
pyLDAvis.save_html(vis1, 'LDA_Vis1.html')

vis2 = pyLDAvis.sklearn.prepare(lda_tfidf, dtm_tfidf, tfidf_vectorizer)
pyLDAvis.save_html(vis2, 'LDA_Vis2.html')

vis3 = pyLDAvis.sklearn.prepare(nmf_tf, dtm_tf, tf_vectorizer)
pyLDAvis.save_html(vis3, 'NMF_Vis1.html')

# vis4 = pyLDAvis.sklearn.prepare(nmf_tfidf, dtm_tfidf, tfidf_vectorizer)
# pyLDAvis.save_html(vis4, 'NMF_Vis2.html')