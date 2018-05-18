# Cryptocurrency Competitor Finder
The world of cryptocurrency extends far beyond not just Bitcoin, Ethereum, and Ripple. As the field of cryptocurrency becomes more popular, people invent more cryptocurrenties. And, it is very important for investers to be discreet at choosing which one would have the best potential value. As there is a program that compares stock prices and values, these investors would need a program that finds cryptocurrency competitors based on their interest. And, our program does not only find competitors from web scrawling, but also recommends the most competitive cryptocurrency out of all the selections.

# Setup
We will be using Python 3 and Python virtual environments:

To make histogram

$ cat myfile.stemmed | perl make_hist.prl

$ python3 -m venv venv

$ . ./venv/bin/activate

$ pip install requests BeautifulSoup4

$ sudo python apt-get python-tk

$ pip install -U gensim

$ pip install -U nltk

$ pip install -U numpy

$ pip install -U scikit-learn

$ python -mpip install -U matplotlib

# Tools
1. web_robot.py: use Python BeautifulSoup to build a robot that crawls https://coinmarketcap.com/ and the official websites of the first 100 cryptocurrencies on CoinMarketCap; write the website contents to myfile.txt.

2. tokenizer.py: use nltk.tokenize package to tokenize the content of myfile.txt; write to myfile.raw.

3. stemmer.py: use nltk.tokenize package to tokenize the content of myfile.txt first and then use nltk Porter Stemmers to stem each token; write to myfile.stemmed.

4. similarity.py: Outputs pairwise cosine similarity in a matrix form and uses (k-mean clustering and agglomerative clustering to separate these data into different clusters

# To generate new data
1. python3 web_robot.py
2. python3 tokenizer.py
3. python3 stemmer.py
4. cat myfile.tokenized | perl make_hist.prl>myfile.tokenized.hist
5. cat myfile.stemmed | perl make_hist.prl>myfile.stemmed.hist


