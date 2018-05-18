import nltk
input_file= open("myfile.txt").read()
output_file=open("myfile.raw", "w")
tokens = nltk.word_tokenize(input_file)
for token in tokens:
    if(token == ".I"):
        output_file.write(token)
        output_file.write(" ")
    else:
        output_file.write(token)
        output_file.write("\n")
