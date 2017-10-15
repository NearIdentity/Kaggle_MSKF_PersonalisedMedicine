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


def create_corpus(dictionary, article_array):
	i_article = 0
	full_tokens_list = []
	for article in article_array:	
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

'''
def generate_X(article_array, num_topics, dictionary, lsi_model):
	i_article = 0
	X_data = np.empty((article_array.shape[0], num_topics))
	for article_text in article_array:
		article_tokens = generate_tokens(article_text)		
		article_vec_bow = dictionary.doc2bow(article_tokens)
		article_vec_lsi = lsi_model[article_vec_bow]
		x_article = [v[1] for v in article_vec_lsi]
		X_data[i_article, :] = np.array(x_article)
		i_article += 1
	return X_data	
'''


def mutation_type(mutation, type0_data, type1_data):
	for type0 in type0_data.keys():
		if mutation in type0_data[type0]:
			return type0
	for type1 in type1_data.keys():
		if mutation in type1_data[type1]:
			return type1
	return "other"


def type0_parameters(type0_data):
	amino_acid_letters = set([])
	simple_changes_first = set([x[0].upper() for x in type0_data["simple"]]) 
	amino_acid_letters = amino_acid_letters.union(simple_changes_first)
	simple_changes_last = set([x[len(x)-1].upper() for x in type0_data["simple"]])
	amino_acid_letters = amino_acid_letters.union(simple_changes_last)
	null_changes_last = set([x[len(x)-1].upper() for x in type0_data["null"]])
	amino_acid_letters = amino_acid_letters.union(null_changes_last)
	degenerate_changes_first = set([x[0].upper() for x in type0_data["degenerate"]])
	amino_acid_letters = amino_acid_letters.union(degenerate_changes_first)
	
	simple_changes_num = set([int(x[1:len(x)-1]) for x in type0_data["simple"]])
	null_changes_num = set([int(x[4:len(x)-1]) for x in type0_data["null"]])
	degenerate_changes_num = set([int(x[1:len(x)-1]) for x in type0_data["degenerate"]])

	max_num = max(max(simple_changes_num), max(null_changes_num), max(degenerate_changes_num))
	position_normaliser = 10.0**np.ceil(np.log(max_num)/np.log(10))

	return amino_acid_letters, position_normaliser		


def generate_X(mutation_array, article_array, type0_data, type1_data, num_topics, dictionary, lsi_model):
	
	num_data = mutation_array.shape[0]
	if (article_array.shape[0] != num_data):
		print "# Error [generate_X(...)]: mutation_array.shape[0] != article_array.shape[0]"
		return None

	type0_amino_acids, type0_normaliser = type0_parameters(type0_data)
	num_type1 = len(type1_data.keys())
	mutation_data_len = (2*len(type0_amino_acids)+1) + num_type1

	i_data = 0
	X_data = np.empty((article_array.shape[0], mutation_data_len + num_topics), dtype=float)
	for article_text in article_array:
		type0_first = np.zeros((1,len(type0_amino_acids)))
		type0_loc = np.zeros((1,1))
		type0_last = np.zeros((1,len(type0_amino_acids)))
		type1_vec = np.zeros(num_type1)

		mutation = mutation_array[i_data]

		change_type = mutation_type(mutation, type0_data, type1_data)

		if(change_type == "simple"):
			amino_acid_first = mutation[0]
			type0_first = np.array([int(x==amino_acid_first) for x in type0_amino_acids], dtype=float)
			amino_acid_last = mutation[len(mutation)-1]
			type0_last = np.array([int(x==amino_acid_last) for x in type0_amino_acids], dtype=float)
			type0_loc[0,0] = int(mutation[1:len(mutation)-1]) / type0_normaliser
		elif(change_type == "null"):
			amino_acid_last = mutation[len(mutation)-1]
			type0_last = np.array([int(x==amino_acid_last) for x in type0_amino_acids], dtype=float)
			type0_loc[0,0] = int(mutation[4:len(mutation)-1]) / type0_normaliser
		elif(change_type == "degenerate"):
			amino_acid_first = mutation[0]
			type0_first = np.array([int(x==amino_acid_first) for x in type0_amino_acids], dtype=float)
			type0_last = np.ones((1,len(type0_amino_acids)))/float(len(type0_amino_acids))
			type0_loc[0,0] = int(mutation[1:len(mutation)-1]) / type0_normaliser
		else:
			type1_vec = np.array([int(category==change_type) for category in type1_data.keys()], dtype=float)
		
		X_data[i_data,0:len(type0_amino_acids)] = type0_first
		X_data[i_data,len(type0_amino_acids):len(type0_amino_acids)+1] = type0_loc
		X_data[i_data,len(type0_amino_acids)+1:2*len(type0_amino_acids)+1] = type0_last
		X_data[i_data,2*len(type0_amino_acids)+1:mutation_data_len] = type1_vec
		
		article_tokens = generate_tokens(article_text)		
		article_vec_bow = dictionary.doc2bow(article_tokens)
		article_vec_lsi = lsi_model[article_vec_bow]
		x_article = [v[1] for v in article_vec_lsi]

		X_data[i_data, mutation_data_len:(mutation_data_len + num_topics)] = np.array(x_article)

		i_data += 1

	return X_data


'''
num_classes = 9
topic_count = 500
num_nodes_layer=512
dropout_value=0.2
num_hidden_layers=3
'''
def nn_model(num_classes=9, num_input_nodes=549, num_nodes_layer=512, dropout_value=0.2, num_hidden_layers=3):
	model = Sequential()
	model.add(Dense(num_nodes_layer, input_dim=num_input_nodes, init='normal', activation='relu'))
	hidden_layer_count = 0
	while(hidden_layer_count<num_hidden_layers):
		model.add(Dropout(dropout_value))
		model.add(Dense(num_nodes_layer, init='normal', activation='relu'))
		hidden_layer_count += 1
	model.add(Dense(num_classes, init='normal', activation="softmax"))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


def generate_mutation_types(variant_dataframe):
	
	variant_dataframe["Simple Variant Status"] = variant_dataframe.Variation.str.contains(r"[A-Z]\d{1,7}[A-Z]$", case=True)
	variant_dataframe["Multi-Source Variant Status"] = variant_dataframe.Variation.str.contains(r"[A-Z]{2,10}\d{1,7}[A-Z]$", case=True)
	variant_dataframe["Degenerate Variant Status"] = variant_dataframe.Variation.str.contains(r"[A-Z]\d{1,7}\*$", case=True)   
	variant_dataframe["Null Variant Status"] = variant_dataframe.Variation.str.contains(r"^null\d{1,7}[A-Z]", case=False)
	variant_dataframe["Fusion Variant Status"] = variant_dataframe.Variation.str.contains(r"^.*Fusion$", case=False)
	variant_dataframe["Splice Variant Status"] = variant_dataframe.Variation.str.contains(r"^.*splice$", case=False)
	variant_dataframe["Trunc Variant Status"] = variant_dataframe.Variation.str.contains(r"^.*trunc$", case=False)
	variant_dataframe["Del Variant Status"] = variant_dataframe.Variation.str.contains(r"^.*del$", case=False)
	variant_dataframe["Dup Variant Status"] = variant_dataframe.Variation.str.contains(r"^.*dup$", case=False)
	variant_dataframe["DelIns Variant Status"] = variant_dataframe.Variation.str.contains(r"^.*delins.*", case=False)
	variant_dataframe["Ins Variant Status"] = variant_dataframe.Variation.str.contains(r"^.*ins.*", case=False)
	
	simple_changes = variant_dataframe
	simple_changes = simple_changes[simple_changes["Simple Variant Status"] == True]
	simple_changes = simple_changes[simple_changes["Multi-Source Variant Status"] == False]["Variation"]
#	multi_source_changes = variant_dataframe[variant_dataframe["Multi-Source Variant Status"] == True]["Variation"]
	degenerate_changes = variant_dataframe[variant_dataframe["Degenerate Variant Status"] == True]["Variation"]
	null_changes = variant_dataframe[variant_dataframe["Null Variant Status"] == True]["Variation"]
	fusion_changes = variant_dataframe[variant_dataframe["Fusion Variant Status"] == True]["Variation"]
	splice_changes = variant_dataframe[variant_dataframe["Splice Variant Status"] == True]["Variation"]
	trunc_changes = variant_dataframe[variant_dataframe["Trunc Variant Status"] == True]["Variation"]
	del_changes = variant_dataframe[variant_dataframe["Del Variant Status"] == True]["Variation"]
	dup_changes = variant_dataframe[variant_dataframe["Dup Variant Status"] == True]["Variation"]
	delins_changes = variant_dataframe[variant_dataframe["DelIns Variant Status"] == True]["Variation"]
	ins_changes = variant_dataframe
	ins_changes = ins_changes[ins_changes["Ins Variant Status"] == True]
	ins_changes = ins_changes[ins_changes["DelIns Variant Status"] == False]["Variation"]
	other_changes = variant_dataframe
	other_changes = other_changes[other_changes["Simple Variant Status"] == False]
	other_changes = other_changes[other_changes["Degenerate Variant Status"] == False]
	other_changes = other_changes[other_changes["Null Variant Status"] == False]
	other_changes = other_changes[other_changes["Fusion Variant Status"] == False]
	other_changes = other_changes[other_changes["Splice Variant Status"] == False]
	other_changes = other_changes[other_changes["Trunc Variant Status"] == False]
	other_changes = other_changes[other_changes["Del Variant Status"] == False]
	other_changes = other_changes[other_changes["Dup Variant Status"] == False]
	other_changes = other_changes[other_changes["Ins Variant Status"] == False]["Variation"]

	change_type0_dict = 	{"simple":set(simple_changes.values), 
#				"multi_source:"set(multi_source_changes.values), 
				"degenerate":set(degenerate_changes.values), 
				"null":set(null_changes.values)}
	change_type1_dict = 	{"fusion":set(fusion_changes.values), 
				"splice":set(splice_changes.values), 
				"trunc":set(trunc_changes.values), 
				"del":set(del_changes.values), 
				"dup":set(dup_changes.values), 
				"delins":set(delins_changes.values), 
				"ins":set(ins_changes.values), 
				"other":set(other_changes.values)}

	return change_type0_dict, change_type1_dict
	
if __name__ == "__main__":

#	num_train_samples = 100
#	num_test_samples = 100

	topic_count = 500

	""" Read Data """
	train_variant = pd.read_csv(getcwd()+"/input/training_variants")
	test_variant = pd.read_csv(getcwd()+"/input/test_variants")
	train_text = pd.read_csv(getcwd()+"/input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
	test_text = pd.read_csv(getcwd()+"/input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

	all_variant = pd.concat([train_variant, test_variant])
	all_genes = all_variant["Gene"]
#	all_changes = all_variant["Variation"]

	change_type0, change_type1 = generate_mutation_types(all_variant)
	num_amino_acids_type0 = len(type0_parameters(change_type0)[0])
	num_type1 = len(change_type1.keys())

	'''
	simple_changes_first = set([x[0].upper() for x in simple_changes.values]) 
	simple_changes_last = set([x[len(x)-1].upper() for x in simple_changes.values])
	null_changes_last = set([x[len(x)-1].upper() for x in null_changes.values])
	degenerate_changes_first = set([x[0].upper() for x in degenerate_changes.values])
	
	simple_changes_num = set([int(x[1:len(x)-1]) for x in simple_changes.values])
	null_changes_num = set([int(x[4:len(x)-1]) for x in null_changes.values])
	degenerate_changes_num = set([int(x[1:len(x)-1]) for x in degenerate_changes.values])
	max_num = max(max(simple_changes_num), max(null_changes_num), max(degenerate_changes_num))
	normaliser = 10.0**np.ceil(np.log(max_num)/np.log(10)) 
	'''

	train_articles = train_text["Text"]
#	train_articles = train_articles[:num_train_samples]

	test_articles = test_text["Text"]
#	test_articles = test_articles[:num_test_samples]

	full_dictionary = create_save_full_dictionary(train_articles, test_articles)
	train_corpus = create_corpus(full_dictionary, train_articles)
	train_lsi = gensim.models.LsiModel(train_corpus, id2word=full_dictionary, num_topics=topic_count)

	train_mutations = train_variant["Variation"]
#	train_mutations = train_mutations[:num_train_samples]

	test_mutations = test_variant["Variation"]
#	test_mutations = test_mutations[:num_test_samples]

#	X_train = generate_X(, topic_count, full_dictionary, train_lsi)
	X_train = generate_X(train_mutations, train_articles, change_type0, change_type1, topic_count, full_dictionary, train_lsi)
#	X_test = generate_X(test_articles, topic_count, full_dictionary, train_lsi)
	X_test = generate_X(test_mutations, test_articles, change_type0, change_type1, topic_count, full_dictionary, train_lsi)
	
	test_index = test_variant['ID'].values

	y_train = train_variant["Class"].values
#	y_train = y_train[:num_train_samples]
	encoder = LabelEncoder()
	encoder.fit(y_train)
	encoded_y = encoder.transform(y_train)
	dummy_y = np_utils.to_categorical(encoded_y)

	estimator = KerasClassifier(build_fn=nn_model, epochs=10, batch_size=64, num_classes=9, num_input_nodes=topic_count+2*num_amino_acids_type0+1+num_type1, num_nodes_layer=512, dropout_value=0.2, num_hidden_layers=3)
	estimator.fit(X_train, dummy_y, validation_split=0.05)
	y_predicted = estimator.predict_proba(X_test)

	""" Submission """
	submission = pd.DataFrame(y_predicted)
	submission['id'] = test_index
	submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']
	submission.to_csv("submission.csv",index=False)

