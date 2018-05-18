from __future__ import print_function
import nltk
from nltk.stem import *
from nltk.stem.porter import *
input_file= open("myfile.txt").read()
output_file=open("myfile.stemmed", "w")
tokens = nltk.word_tokenize(input_file)
stemmer = PorterStemmer()
for token in tokens:
    if(token == ".I"):
        output_file.write(stemmer.stem(token))
        output_file.write(" ")
    else:
        output_file.write(stemmer.stem(token))
        output_file.write("\n")
output_file.write(".I 0")
