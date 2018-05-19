import gensim
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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

def initialize_gui():
	window = Tk()
	window.title("Cryptocurrency Compatitor Detector")
	window.geometry('600x800')
	buttons = ['Show Cluster', 'Calculate the Best Competitor']
	lb = Label(window, text="1")
	for i in range(2):
		l = tk.Label(window,
					text=buttons[i],
					)
		l.place(x = 20, y = 30 + i * 30, width=120, height=25)
	window.mainloop()

# read myfile to create a model 
# has to read 
# raw-documents 
sentiment_dic = {}
docs_vector = {}
gen_docs = []
cluster_group = {}
raw_documents = []
list_names = []
name = ""
temp_txt = ""
first = 1
second = -1

# in order to get rid of extermely sparse matrices
threshold_value = 18
sid = SentimentIntensityAnalyzer()
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
					sentiment_dic[name] = sid.polarity_scores(temp_txt)
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
	np.set_printoptions(threshold=np.nan)

	while(1):
		print ("*****************************************************************")
		print ("*            Cryptocurrency Analysis Application                *")
		print ("*                                                               *")
		print ("*                                                               *")
		print ("*                                                               *")
		print ("*     1: Compute cosine-similarity between all possible pairs   *")
		print ("*     2: Categorize and Visualize K-Mean Cluster                *")
		print ("*     3: Categorize and Visualize Hierarchial Cluster           *")
		print ("*     4: The Best Competitor in K-Mean Cluster                  *")
		print ("*     5: The Best Competitor in Hierarchial Cluster             *")
		print ("*     6: Exit                                                   *")
		print ("*****************************************************************")
		choice = input ("Enter your choice (1 ~ 6): ")
		cluster_group = {}
		if(choice == 1):
	   		print (pairwise_similarity)
		if(choice == 2):
			k_value = input ("Enter k value: ")
			kmeans = KMeans(n_clusters=k_value, init='k-means++').fit_predict(pairwise_similarity)
			kmeans_f = KMeans(n_clusters=k_value, init='k-means++').fit(pairwise_similarity)
			j = 0
			for i in kmeans:
				k = i.astype(int)
				print (list_names[j])
				print (i)
				print (" ")
				j = j + 1

			plt.xlabel('X')
			plt.ylabel('Y')
			plt.scatter(pairwise_similarity[:,0], pairwise_similarity[:,1],
				c=kmeans_f.labels_, edgecolor='')
			plt.show()

		if(choice == 3):
			n_value = input ("Number of cluster: ")
			hcluster = AgglomerativeClustering(n_clusters=n_value).fit_predict(pairwise_similarity)
			hcluster_f = AgglomerativeClustering(n_clusters=n_value).fit(pairwise_similarity)

		
			j = 0
			for i in hcluster:
				k = i.astype(int)
				print (list_names[j])
				print (i)
				print (" ")
				j = j + 1

			plt.xlabel('X')
			plt.ylabel('Y')
			plt.scatter(pairwise_similarity[:,0], pairwise_similarity[:,1],
				c=hcluster_f.labels_, edgecolor='')
			plt.show()
		if(choice == 4):
			k_value = input ("Enter k value: ")
			kmeans = KMeans(n_clusters=k_value, init='k-means++').fit_predict(pairwise_similarity)
			kmeans_f = KMeans(n_clusters=k_value, init='k-means++').fit(pairwise_similarity)

			for i in kmeans:
				k = i.astype(int)
				cluster_group[k] = list()

			j = 0
			for i in kmeans:
				k = i.astype(int)
				cluster_group[k].append(list_names[j])
				j = j + 1

			max_name = ""
			first = -1
			for i in range(len(cluster_group)):
				for name in cluster_group[i]:
					if(first == -1):
						max_name = cluster_group[i][0]
						max_val = sentiment_dic[max_name]["pos"] - sentiment_dic[max_name]["neg"]
						first = 1

					val = sentiment_dic[name]["pos"] - sentiment_dic[name]["neg"]
					if(max_val < val):
						max_val = val
						max_name = name
				first = -1

				print ("")
				print ("The Cluster No:")
				print (i) 
				print ("The Best Competitor:")
				print (max_name)

		if(choice == 5):
			n_value = input ("Number of cluster: ")
			hcluster = AgglomerativeClustering(n_clusters=n_value).fit_predict(pairwise_similarity)
			hcluster_f = AgglomerativeClustering(n_clusters=n_value).fit(pairwise_similarity)



			for i in hcluster:
				k = i.astype(int)
				cluster_group[k] = list()

			j = 0
			for i in hcluster:
				k = i.astype(int)
				cluster_group[k].append(list_names[j])
				j = j + 1

			max_name = ""
			first = -1
			for i in range(len(cluster_group)):
				for name in cluster_group[i]:
					if(first == -1):
						max_name = cluster_group[i][0]
						max_val = sentiment_dic[max_name]["pos"] - sentiment_dic[max_name]["neg"]
						first = 1

					val = sentiment_dic[name]["pos"] - sentiment_dic[name]["neg"]
					if(max_val < val):
						max_val = val
						max_name = name
				first = -1


				print ("")
				print ("The Cluster No:")
				print (i) 
				print ("The Best Competitor:")
				print (max_name)
		if (choice == 6):
				sys.exit()

	f.close()