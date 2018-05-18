import nltk
input_file= open("myfile.txt").read()
output_file=open("myfile.tokenized", "w")
tokens = nltk.word_tokenize(input_file)
for token in tokens:
    if(token == ".I"):
        output_file.write(token)
        output_file.write(" ")
    else:
        output_file.write(token)
        output_file.write("\n")
output_file.write(".I 0")
