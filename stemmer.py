from __future__ import print_function
import nltk
from nltk.stem import *
from nltk.stem.porter import *
input_file= open("myfile.txt").read()
output_file=open("myfile.stemmed", "w")
pre = None;
index = 0;
tokens = nltk.word_tokenize(input_file)
stemmer = PorterStemmer()
for token in tokens:
    if(token == ".I"): # if the token is ".I", start of a new document; ends with space.
        if(pre != ".I"): # if the 2nd pre is not .I then it is okay
            output_file.write(stemmer.stem(token))
            output_file.write(" ")
            pre = ".I"
    else:
        if(pre == ".I"): # if the pre is ".I", check index
            output_file.write(stemmer.stem(token)) # cryptocurrency name
            output_file.write("\n")
            index = index + 1
            if(index == 2):
                pre = token
        else:
            output_file.write(stemmer.stem(token))
            output_file.write("\n")
            pre = token
output_file.write(".I 0")
