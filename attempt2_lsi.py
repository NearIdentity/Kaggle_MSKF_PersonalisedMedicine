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
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

""" Ancilliae for Text-Parsing Functions Below """
tokenizer = RegexpTokenizer(r'\w+')
stop_list = stopwords.words('english')
p_stemmer = PorterStemmer()

""" Gensim Data Dictionary from Training and Test Sets """
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

""" Gensim Corpus from Integrated Dictionary and Article """
def create_corpus(dictionary, article_array):
	i_article = 0
	full_tokens_list = []
	for article in article_array:	
		parsed_tokens = generate_tokens(article)
		full_tokens_list.append(parsed_tokens)
		print "# Diagnostic: Article "+str("%04d" %i_article)+" parsed for addition to merged corpus"
		i_article += 1
	corpus = [dictionary.doc2bow([entry for article_tokens in full_tokens_list for entry in article_tokens])]
	return corpus

""" Tokeniser for Textual Data """
def generate_tokens(article):
	raw_data = article.lower()
	raw_data = unicode(raw_data, errors='ignore')

	raw_tokens = tokenizer.tokenize(raw_data)

	stopped_tokens = [t for t in raw_tokens if t not in stop_list]

	return [p_stemmer.stem(s) for s in stopped_tokens]

""" Search Function for Mutation Type Given Loaded \'dict\' Objects Containing Relevant Data """
def mutation_type(mutation, type0_data, type1_data):
	for type0 in type0_data.keys():
		if mutation in type0_data[type0]:
			return type0
	for type1 in type1_data.keys():
		if mutation in type1_data[type1]:
			return type1
	return "other"

""" Amino Acid Alphabet and Maximum Position Value for Mutaions of the Form [AminoAcidLetter0][PositionNumber][AminoAcidLetter1] (Type0) """
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

""" Neural Network Input Data Matrix from Mutation and Textual Data Along with Relevant Gensim Objects """

""" 	Each row being of the form: [1-hot vector for type0 amino acid (a0-many values), normalised type0 position (1 value), 1-hot vector for type0 amino acid (a0-many values), 1-hot vector for type1 mutations (n1-many values), topic model projections from article (t-many values)] 
	Each row having (a0 + 1 + a0 + n1 + t) elements for every data sample """
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

		if(change_type == "simple"):	# Type0 Mutation of Form [AminoAcid0][Position][AminoAcid1] 
			amino_acid_first = mutation[0]
			type0_first = np.array([int(x==amino_acid_first) for x in type0_amino_acids], dtype=float)	# One-hot vector: AminoAcid0
			amino_acid_last = mutation[len(mutation)-1]
			type0_last = np.array([int(x==amino_acid_last) for x in type0_amino_acids], dtype=float)	# One hot vector: AminoAcid1
			type0_loc[0,0] = int(mutation[1:len(mutation)-1]) / type0_normaliser	# Normalised position
		elif(change_type == "null"):	# Type0 Mutation of Form null[Position][AminoAcid1] -- Zero-Hot Vector for \'null\' Value
			amino_acid_last = mutation[len(mutation)-1]
			type0_last = np.array([int(x==amino_acid_last) for x in type0_amino_acids], dtype=float)	# One hot vector: AminoAcid1
			type0_loc[0,0] = int(mutation[4:len(mutation)-1]) / type0_normaliser	# Normalised position
		elif(change_type == "degenerate"):	# Type0 Mutation of Form [AminoAcid0][Position]* -- Wildcard for AminoAcid1 => Equiprobable Vector Instead of One-Hot
			amino_acid_first = mutation[0]	
			type0_first = np.array([int(x==amino_acid_first) for x in type0_amino_acids], dtype=float)	# One-hot vector: AminoAcid0
			type0_last = np.ones((1,len(type0_amino_acids)))/float(len(type0_amino_acids))	# Equiprobable vector: AminoAcid1; all values equally likely
			type0_loc[0,0] = int(mutation[1:len(mutation)-1]) / type0_normaliser	# Normalised position
		else:	# Type1 Mutation
			type1_vec = np.array([int(category==change_type) for category in type1_data.keys()], dtype=float)
		
		""" Mutation Data at Start of Row  """
		X_data[i_data,0:len(type0_amino_acids)] = type0_first
		X_data[i_data,len(type0_amino_acids):len(type0_amino_acids)+1] = type0_loc
		X_data[i_data,len(type0_amino_acids)+1:2*len(type0_amino_acids)+1] = type0_last
		X_data[i_data,2*len(type0_amino_acids)+1:mutation_data_len] = type1_vec
		
		""" Projection of Textual Data onto Topic Model Basis Set """
		article_tokens = generate_tokens(article_text)		
		article_vec_bow = dictionary.doc2bow(article_tokens)
		article_vec_lsi = lsi_model[article_vec_bow]
		x_article = [v[1] for v in article_vec_lsi]

		""" Textual Projection Data at End of Row """
		X_data[i_data, mutation_data_len:(mutation_data_len + num_topics)] = np.array(x_article)

		i_data += 1

	return X_data

""" Traditional Multi-Layer Deep Neural Network Model to be Used as Classifier """
def nn_model_traditional(num_classes=9, num_input_nodes=549, num_nodes_layer=512, dropout_value=0.2, num_hidden_layers=3):
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

""" Convolutional Neural Network Model to be Used as Classifier """
def nn_model_convolutional(num_filters=50, len_filter=10, num_input_nodes=549, num_classes=9):
	model = Sequential()
	model.add(Convolution1D(nb_filter=num_filters, filter_length=len_filter, activation='relu', input_shape=(num_input_nodes,1)))
	model.add(MaxPooling1D())     # Downsample the output of convolution by 2X.
	model.add(Convolution1D(nb_filter=num_filters, filter_length=len_filter, activation='relu'))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(num_classes, init='normal', activation='softmax'))     # For binary classification, change the activation to 'sigmoid'    	
	model.compile(loss='mse', optimizer='adam', metrics=['mae'])
	return model

""" Collating Mutation Data into \'dict\' Objects of Pre-Determined Types -- Using Manual Observations to Classify Mutations by RegEx Analysis """
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


	topic_count = 4000

	""" Reading Data from Training and Test Sets [Stage2_Training_Set = Stage1_Training_Set + Stage1_Validated_Test_Set] """
	train_variant0 = pd.read_csv(getcwd()+"/input/training_variants")
	train_variant1 = pd.read_csv(getcwd()+"/input/test_variants")
	stage1_results_dataframe = pd.read_csv(getcwd()+"/input/stage1_solution_filtered.csv")
	stage1_indices = list(stage1_results_dataframe["ID"])
	train_variant1 = train_variant1[train_variant1["ID"].isin(stage1_indices)]
	train_variant1["ID"] = train_variant1["ID"] + train_variant0.values.shape[0]	# Training set index offsets to avoid errors in mutation_type(...) due to concatenated data-frames 
	train_variant = pd.concat([train_variant0, train_variant1])
	test_variant = pd.read_csv(getcwd()+"/input/stage2_test_variants.csv")
	train_text0 = pd.read_csv(getcwd()+"/input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
	train_text1 = pd.read_csv(getcwd()+"/input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
	train_text1 = train_text1[train_text1["ID"].isin(stage1_indices)]
	train_text1["ID"] = train_text1["ID"] + train_text0.values.shape[0]	# Training set index offsets to avoid errors in mutation_type(...) due to concatenated data-frames 
	train_text = pd.concat([train_text0, train_text1])
	test_text = pd.read_csv(getcwd()+"/input/stage2_test_text.csv", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

	""" Combined Mutations for Both Training and Test Sets """		
	all_variant = pd.concat([train_variant, test_variant])
	all_genes = all_variant["Gene"]

	""" Compiling Mutation Types and Amino Acids """
	change_type0, change_type1 = generate_mutation_types(all_variant)
	num_amino_acids_type0 = len(type0_parameters(change_type0)[0])
	num_type1 = len(change_type1.keys())


	""" Textual Evidence Data """	
	train_articles = np.array(train_text["Text"], dtype=str)
	test_articles = np.array(test_text["Text"], dtype=str)

	""" Gensim Objects and Models """
	full_dictionary = create_save_full_dictionary(train_articles, test_articles)
	train_corpus = create_corpus(full_dictionary, train_articles)	# Only training set articles used as basis for topic model to search test data against
	train_lsi = gensim.models.LsiModel(train_corpus, id2word=full_dictionary, num_topics=topic_count)

	""" Mutation Datasets """
	train_mutations = np.array(train_variant["Variation"], dtype=str)
	test_mutations = np.array(test_variant["Variation"], dtype=str)

	""" Input Data for Neural Networks to be Used """
	X_train = generate_X(train_mutations, train_articles, change_type0, change_type1, topic_count, full_dictionary, train_lsi)
	X_test = generate_X(test_mutations, test_articles, change_type0, change_type1, topic_count, full_dictionary, train_lsi)

	""" Saving Input Data for Future Use if Necessary """	
	X_train_dataframe = pd.DataFrame(X_train)
	X_train_dataframe.to_csv("X_train.csv", index=False)
	X_test_dataframe = pd.DataFrame(X_test)
	X_test_dataframe.to_csv("X_test.csv", index=False)

	""" Indices for Test Data Output """	
	test_index = test_variant['ID'].values

	""" Labels (in One-Hot Vector Form) for Stage 1 Training Data and Validated Stage 1 Test Data """
	y_train0 = train_variant0["Class"].values
	encoder = LabelEncoder()
	encoder.fit(y_train0)
	encoded_y0 = encoder.transform(y_train0)
	dummy_y0 = np_utils.to_categorical(encoded_y0)
	dummy_y1 = stage1_results_dataframe.values[:,1:]

	if (dummy_y0.shape[1] == dummy_y1.shape[1]):	# Neural Network Processing Only if Stage 1 Training Data Labels and Stage 1 Vaidated Test Data Labels Mutually Commpatible
		""" Consolidated Training Labels for Stage 2"""
		dummy_y = np.empty((dummy_y0.shape[0] + dummy_y1.shape[0], dummy_y1.shape[1]))
		dummy_y[0:dummy_y0.shape[0],:] = dummy_y0
		dummy_y[dummy_y0.shape[0]:dummy_y.shape[0],:] = dummy_y1

		""" Traditional Neural Network Model for Data """
		estimator_trad = KerasClassifier(build_fn=nn_model_traditional, epochs=10, batch_size=64, num_classes=9, num_input_nodes=topic_count+2*num_amino_acids_type0+1+num_type1, num_nodes_layer=1024, dropout_value=0.2, num_hidden_layers=5)
		estimator_trad.fit(X_train, dummy_y, validation_split=0.05)
		y_predicted_trad = estimator_trad.predict_proba(X_test)

		""" Submission File for Traditional Neural Network """
		submission_trad = pd.DataFrame(y_predicted_trad)
		submission_trad['id'] = test_index
		submission_trad.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']
		submission_trad.to_csv("submission_trad.csv",index=False)

		""" Convolutional Neural Network Model for Data """
		estimator_conv = KerasClassifier(build_fn=nn_model_convolutional, epochs=10, batch_size=64, num_classes=9, num_input_nodes=topic_count+2*num_amino_acids_type0+1+num_type1, num_filters=100, len_filter=3)
		estimator_conv.fit(np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)), dummy_y, validation_split=0.05)
		y_predicted_conv = estimator_conv.predict_proba(np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)))

		""" Submission File for Convolutional Neural Network """
		submission_conv = pd.DataFrame(y_predicted_conv)
		submission_conv['id'] = test_index
		submission_conv.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']
		submission_conv.to_csv("submission_conv.csv",index=False)
	else:	# Data Labels for the Two Separate Parts of Stage 2 Training Set Mutually Incompatible -- Not Possible to Train Neural Network
		print "# Error: Shape mismatch for \'dummy_y\' variable to be used for fitting models"

