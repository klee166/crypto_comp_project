import nltk
input_file= open("myfile.txt").read()
output_file=open("myfile.tokenized", "w")
tokens = nltk.word_tokenize(input_file)
index = 0;
skip = False;
skiptwo = False;
connect = False;
for token in tokens:
    if(token == ".I"):
        if(tokens[index+2] == ".I"):
            skip = True
        else:
            if((tokens[index+1] == "Bitcoin"
            and (tokens[index+2] == "Cash" or tokens[index+2] == "Gold" or tokens[index+2] == "Diamond" or tokens[index+2] == "Private"))
            or (tokens[index+1] == "Ethereum" and tokens[index+2] == "Classic")):
                connect = True
                if(tokens[index+3] == ".I"):
                    skip = True
                    skiptwo = True
            output_file.write(token)
            output_file.write(" ")
    else:
        if(skip == True):
            skip = False
        if(skiptwo == True):
            skiptwo = False
        elif(connect == True):
            output_file.write(token)
            output_file.write(" ")
            connect = False
        else:
            output_file.write(token) # cryptocurrency name
            output_file.write("\n")
    index = index + 1
output_file.write(".I 0")
