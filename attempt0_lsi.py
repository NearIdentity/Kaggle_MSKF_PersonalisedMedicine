from sklearn import *
import pandas as pd
import numpy as np
from os import getcwd
import string
import re

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers import Dropout

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

'''
train_genes = set(train_variant["Gene"])
train_genes = [e.lower() for e in train_genes]
train_mutations = set(train_variant["Variation"])
train_mutations = [e.lower() for e in train_mutations]
test_genes = set(test_variant["Gene"])
test_genes = [e.lower() for e in test_genes]
test_mutations = set(test_variant["Variation"])
test_mutations = [e.lower() for e in test_mutations]
'''

tokenizer = RegexpTokenizer(r'\w+')
stop_list = stopwords.words('english')
p_stemmer = PorterStemmer()

def create_save_full_dictionary(train_set, test_set):
	dictionary = gensim.corpora.Dictionary()
	i_train = 0
	for article in train_set:
		parsed_tokens_list = []
		parsed_tokens = generate_tokens(article)
		parsed_tokens_list.append(parsed_tokens)
		dictionary.add_documents(parsed_tokens_list)	
		print "# Diagnostic: Training sample "+str("%04d" %i_train)+" added to dictionary"
		i_train += 1
	i_test = 0
	for article in test_set:
		parsed_tokens_list = []
		parsed_tokens = generate_tokens(article)
		parsed_tokens_list.append(parsed_tokens)
		dictionary.add_documents(parsed_tokens_list)
		print "# Diagnostic: Test sample "+str("%04d" %i_test)+" added to dictionary"
		i_test += 1
	dictionary.save("full_dictionary_lsi")
	return dictionary	

def create_corpus(dictionary, article_set):
	i_article = 0
	full_tokens_list = []
	for article in article_set:	
		parsed_tokens = generate_tokens(article)
		full_tokens_list.append(parsed_tokens)
		print "# Diagnostic: Article "+str("%04d" %i_article)+" parsed for addition to merged corpus"
		i_article += 1
#	print parsed_tokens
#	dictionary = gensim.corpora.Dictionary(parsed_tokens_list)
	corpus = [dictionary.doc2bow([entry for article_tokens in full_tokens_list for entry in article_tokens])]
#	corpus = [dictionary.doc2bow(parsed_tokens)]
#	corpus.save("corpus_lsi")
	return corpus

#def generate_article_tokens(current_article, genes_list, mutations_list, this_gene, this_mutation):
def generate_tokens(article):
	raw_data = article.lower()
#	tmp_genes_list = genes_list
#	tmp_genes_list.remove(this_gene)
#	tmp_mutations_list = mutations_list
#	tmp_mutations_list.remove(this_mutation)
#	raw_data = re.sub(this_gene, "this93n3", raw_data)
#	raw_data = re.sub("[^a-z ^0-9 ^A-Z]|".join(tmp_genes_list), "other93n3", raw_data)
#	raw_data = re.sub(this_mutation, "thismut4t10n", raw_data)
#	raw_data = re.sub("[^a-z ^0-9 ^A-Z]|".join(tmp_mutations_list), "othermut4t10n", raw_data)
	raw_data = unicode(raw_data, errors='ignore')

	raw_tokens = tokenizer.tokenize(raw_data)

	stopped_tokens = [t for t in raw_tokens if t not in stop_list]

	return [p_stemmer.stem(s) for s in stopped_tokens]

def generate_X(article_set, num_topics, dictionary, lsi_model):
	i_article = 0
	X_data = np.empty((article_set.shape[0], num_topics))
	for article_text in article_set:
		article_tokens = generate_tokens(article_text)		
		article_vec_bow = dictionary.doc2bow(article_tokens)
		article_vec_lsi = lsi_model[article_vec_bow]
		x_article = [v[1] for v in article_vec_lsi]
		X_data[i_article, :] = np.array(x_article)
		i_article += 1
	return X_data	

num_classes = 9
topic_count = 500
num_nodes_layer=512
dropout_value=0.2
num_hidden_layers=3

def nn_model():
	model = Sequential()
	model.add(Dense(num_nodes_layer, input_dim=topic_count, init='normal', activation='relu'))
	hidden_layer_count = 0
	while(hidden_layer_count<num_hidden_layers):
		model.add(Dropout(dropout_value))
		model.add(Dense(num_nodes_layer, init='normal', activation='relu'))
		hidden_layer_count += 1
	model.add(Dense(num_classes, init='normal', activation="softmax"))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

	
if __name__ == "__main__":

#	num_train_samples = 100
#	num_test_samples = 100

	""" Read Data """
	train_variant = pd.read_csv(getcwd()+"/input/training_variants")
	test_variant = pd.read_csv(getcwd()+"/input/test_variants")
	train_text = pd.read_csv(getcwd()+"/input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
	test_text = pd.read_csv(getcwd()+"/input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

	#print(train_text.head())
	train_articles = train_text["Text"]
	#train_articles = train_articles[:num_train_samples]

	#print(test_text.head())
	test_articles = test_text["Text"]
	#test_articles = test_articles[:num_test_samples]

	full_dictionary = create_save_full_dictionary(train_articles, test_articles)
	train_corpus = create_corpus(full_dictionary, train_articles)
	train_lsi = gensim.models.LsiModel(train_corpus, id2word=full_dictionary, num_topics=topic_count)

	X_train = generate_X(train_articles, topic_count, full_dictionary, train_lsi)
	X_test = generate_X(test_articles, topic_count, full_dictionary, train_lsi)
	test_index = test_variant['ID'].values

	y_train = train_variant["Class"].values
	#y_train = y_train[:num_train_samples]
	encoder = LabelEncoder()
	encoder.fit(y_train)
	encoded_y = encoder.transform(y_train)
	dummy_y = np_utils.to_categorical(encoded_y)

	estimator = KerasClassifier(build_fn=nn_model, epochs=10, batch_size=64)
	estimator.fit(X_train, dummy_y, validation_split=0.05)
	y_predicted = estimator.predict_proba(X_test)

	""" Submission """
	submission = pd.DataFrame(y_predicted)
	submission['id'] = test_index
	submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']
	submission.to_csv("submission.csv",index=False)

