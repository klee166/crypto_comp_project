import gensim
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import string

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
printable = set(string.printable)

def stem_tk(tks):
	return [stemmer.stem(item) for item in tks]

def normalize(txt):
	return stem_tk(nltk.word_tokenize(txt.lower().translate(remove_punctuation_map)))



# read myfile to create a model 
# has to read 
# raw-documents 
docs_vector = {}
gen_docs = []
raw_documents = []
name = ""
temp_txt = ""
first = -1
#print("Loading the Program... ")
with open('myfile.txt','rb') as f:
	read_txt = f.readlines()
	for text in read_txt:
		text = text.decode('utf-8')
		temp_temp_txt = text.rstrip()
		if(".I" == temp_temp_txt[0:2]):
			if(first == 1):
				#print temp_txt
				raw_documents.append(temp_txt)
			if(len(text) > 1):
				temp_txt = temp_txt + text[2:]
				first = 1
		else:
			temp_txt = temp_txt + text
			first = 1
	#print raw_documents
	vect = TfidfVectorizer(tokenizer=normalize, stop_words='english').fit_transform(raw_documents)
	pairwise_similarity = ((vect * vect.T).A)
	np.set_printoptions(threshold=np.nan)
f.close()

#sims = gensim.similarities.Similarity()
#print(corpus)