import gensim
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import string

# Coming up with stemmer
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
list_names = []
name = ""
temp_txt = ""
first = 1
second = -1

# in order to get rid of extermely sparse matrices
threshold_value = 18
#print("Loading the Program... ")
with open('myfile.txt','rb') as f:
	read_txt = f.readlines()
	for text in read_txt:
		text = text.decode('utf-8')
		temp_temp_txt = text.rstrip()
		if(".I" == temp_temp_txt[0:2]):
			if(first == -1):
				if(len(temp_txt) > threshold_value):
					raw_documents.append(temp_txt)
				else:
					list_names.remove(name)
				temp_txt = ""
			first = -1
		
			temp_temp_temp_txt = temp_temp_txt.split(" ")
			name = temp_temp_temp_txt[1]
			temp_txt = temp_txt + text[3:]
			list_names.append(name)
		else: 
			temp_txt = temp_txt + text


	vect = TfidfVectorizer(tokenizer=normalize, stop_words='english').fit_transform(raw_documents)
	pairwise_similarity = cosine_similarity(vect)

	# predict k_means
	#eigen_values, eigen_vectors = np.linalg.eigh(pairwise_similarity)
	kmeans = KMeans(n_clusters=5, init='k-means++').fit_predict(pairwise_similarity)
	kmeans_f = KMeans(n_clusters=5, init='k-means++').fit(pairwise_similarity)



	hcluster = AgglomerativeClustering(n_clusters=5).fit_predict(pairwise_similarity)
	hcluster_f = AgglomerativeClustering(n_clusters=5).fit(pairwise_similarity)

	# categorize and come up with k means
	print ("K-Mean Clustering")
	j = 0
	for i in np.nditer(kmeans):
		print list_names[j]
		print i
		j = j + 1

	j = 0
	print("Hierachy Clustering")
	for i in np.nditer(hcluster):
		print list_names[j]
		print i
		j = j + 1



	plt.xlabel('X')
	plt.ylabel('Y')
	plt.scatter(pairwise_similarity[:,0], pairwise_similarity[:,1],
		c=kmeans_f.labels_, edgecolor='')
	plt.show()
	np.set_printoptions(threshold=np.nan)
   	print pairwise_similarity
	f.close()
