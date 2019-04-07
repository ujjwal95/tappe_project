import os
import gc
import csv
import sys
import operator
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import torch
# from pytorch_pretrained_bert import BertModel
import matplotlib.pyplot as plt
sys.path.append('/Users/karanjani/Desktop/InferSent-master/encoder') #Append path to Infersent models
sys.path.append('/Users/karanjani/Desktop/gensen') #Append path to GenSen models
sys.path.append('/Users/karanjani/Desktop/bert') #Append path to BERT service
from service.client import BertClient
from gensen import GenSen, GenSenSingle
from models import InferSent
from glob import glob
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

files = glob("/Users/karanjani/Desktop/ab_003_full.csv")

for file in files:
	df = pd.read_csv(file)

	#extracting list of utterances from CSV and cleaning them 
	messageString = df['stringList'].tolist()
	stopwords = ['um','a','the','uh','an']
	cleanedStrings = []

	#removes stopwords and stuttered repeats i.e. "okay okay how are you doing" --> "okay how are you doing"
	for string in messageString:
		cleanTemp = [word for word in string.lower().split(" ") if word not in stopwords]
		if len(cleanTemp) > 2:
			for i in range(len(cleanTemp)-2): 
				if cleanTemp[i] == cleanTemp[i+1]:
					cleanTemp[i] = 0
		elif (len(cleanTemp) == 2) and (cleanTemp[0] == cleanTemp[1]):
			cleanTemp[0] = 0

		cleanTemp = [word for word in cleanTemp if word != 0] #actually removes duplicates that have been reassigned to 0

		#Instantiates client for cloud NLP to lemmatize each word using google cloud to speech:
		# client = language.LanguageServiceClient()
		# tempString = ' '.join(cleanTemp) #creates temporary string that will undergo lemmatization
		# text = tempString
		# document = types.Document(content=text, type=enums.Document.Type.PLAIN_TEXT)
		# tokens = client.analyze_syntax(document).tokens
		# cleanTemp = [token.lemma for token in tokens]

		cleanedStrings.append(' '.join(cleanTemp))

	###############################
	###### BERT EMBEDDINGS #######
	###############################

	# ec = BertClient()
	# bert_embeddings = ec.encode(cleanedStrings)

	# print bert_embeddings


	###############################
	#### FACEBOOK EMBEDDINGS #####
	###############################
	# V = 2
	# MODEL_PATH = '/Users/karanjani/Desktop/InferSent-master/encoder/infersent%s.pkl' % V
	# params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, 'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
	# infersent = InferSent(params_model)
	# infersent.load_state_dict(torch.load(MODEL_PATH))

	# W2V_PATH = '/Users/karanjani/Desktop/InferSent-master/dataset/fastText/crawl-300d-2M.vec' #ENTER PATH TO FASTTEXT
	# infersent.set_w2v_path(W2V_PATH)

	# infersent.build_vocab(cleanedStrings, tokenize=True)
	# embeddings = infersent.encode(cleanedStrings, tokenize=True)

	# fbvecFrame = pd.DataFrame(list(embeddings)) #converting Facebook embeddings tuple to dataframe
	# FBcols = ["FB%d" % d for d in range(4096)] #creating list of column names for Facebook vectors
	# fbvecFrame.columns = FBcols #reset column names to be FB1, FB2 ... FB4096 
	# fullFrame = pd.concat([df, fbvecFrame], axis=1) #creating new dataframe with Facebook vectors

	################################
	###### GENSEN EMBEDDINGS ######
	################################
	gensen_1 = GenSenSingle(model_folder='/Users/karanjani/Desktop/gensen/data/models',filename_prefix='nli_large_bothskip',pretrained_emb='/Users/karanjani/Desktop/gensen/data/embedding/glove.840B.300d.h5')
	gensen_2 = GenSenSingle(model_folder='/Users/karanjani/Desktop/gensen/data/models',filename_prefix='nli_large_bothskip_parse',pretrained_emb='/Users/karanjani/Desktop/gensen/data/embedding/glove.840B.300d.h5')
	# reps_h, reps_h_t = gensen_1.get_representation(messageString, pool='last', return_numpy=True, tokenize=True)
	gensen = GenSen(gensen_1, gensen_2)
	reps_h, reps_h_t = gensen.get_representation(cleanedStrings, pool='last', return_numpy=True, tokenize=True)
	

	gsvecFrame = pd.DataFrame(reps_h_t)
	GScols = ["GS%d" % d for d in range(4096)]
	gsvecFrame.columns = GScols
	# fullFrame = pd.concat([fullFrame, gsvecFrame], axis=1)
	fullFrame = pd.concat([df, gsvecFrame], axis=1)

	################################
	###### GOOGLE EMBEDDINGS ######
	################################
	# googModule = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

	# with tf.Session() as session:
	# 	session.run([tf.global_variables_initializer(), tf.tables_initializer()])
	# 	GGembeddings = session.run(googModule(cleanedStrings))

	# ggvecFrame = pd.DataFrame(np.array(GGembeddings))
	# GGcols = ["GG%d" % d for d in range(512)]
	# ggvecFrame.columns = GGcols
	# fullFrame = pd.concat([fullFrame, ggvecFrame], axis=1)

	################################
	####### ELMO EMBEDDINGS #######
	################################
	# elmoModule = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
	# with tf.Session() as session:
	# 	session.run([tf.global_variables_initializer(), tf.tables_initializer()])
	# 	elmoembeddings = session.run(elmoModule(cleanedStrings, signature="default"))

	# elmovecFrame = pd.DataFrame(np.array(elmoembeddings))
	# elmocols = ["ELMO%d" % d for d in range(1024)]
	# elmovecFrame.columns = elmocols
	# fullFrame = pd.concat([fullFrame, elmovecFrame], axis=1)

	###################################
	baseName = file.split('/')[-1].replace(".csv",'')
	newName = baseName + "withVecs" + ".csv"

	fullFrame.to_csv(newName, index=False)
	#Clear system memory for future processing by deleting, garbage collecting, and setting to null 
	del df
	del fullFrame
	# del fbvecFrame
	del gsvecFrame
	# del ggvecFrame
	# del elmovecFrame
	gc.collect()
	df = pd.DataFrame()
	fullFrame = pd.DataFrame()
	# fbvecFrame = pd.DataFrame()
	gsvecFrame = pd.DataFrame()
	# ggvecFrame = pd.DataFrame()
	# elmovecFrame = pd.DataFrame()







