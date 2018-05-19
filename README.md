# Cryptocurrency Analysis Engine with IR technology
There are more than 1565 cryptocurrencies over the internet and the number is growing. Different cryptocurrencies are designed and implemented based on different technology (e.g. Proof-of-Work, Proof-of-Stake, Zero-knowledge, and etc.), and they are aimed to tackle different problems. Individual investors, cryptocurrency enthusiasts, venture capitalists and tech companies all try to analyze and evaluate the cryptocurrencies based on various dimensions. In this project, we implement an alternative way to analyze cryptocurrencies using IR (Information Retrieval technology).

We crawl texts regarding around 100 cryptocurrencies from https://coinmarketcap.com/ and Bitcoin Wiki using Python robots and vectorize them using Vector IR Model implemented in Perl. We measure the cosine similarity between the vector representations of different cryptocurrencies and identify the groups of cryptocurrencies that target at similar topics and compete with each other. Another way to determine competitor groups is to apply K-Means clustering and Hierarchical Agglomerative Clustering to this large text datasets we have collected, and then use Python library to do sentiments analysis on the texts. We pick the best competitor within a certain cluster in terms of positive/negative sentiments scores.

In addition, we build a Naive Bayesian Model and train the model using training.txt based on the classification dimensions provided by https://medium.com/swlh/a-better-taxonomy-for-cryptocurrencies-cbffd2e1b58c: 1. Mode of Payment/Currency 2. Store of Value 3. Protocol Improvement 4. Coin-as-a-Service 5. Utility Token. We then run the Bayesian model on the test sets--myfile.txt and return the log likelihood for each category, by which we classify each cryptocurrency into one or two categories.

Our User Interface for Perl:
============================================================
==     Welcome to the 600.466 Cryptocurrency Analysis Engine
==
      == Total Documents: 91
============================================================

OPTIONS:
  1 = Find the similarity between two cryptocurrencies
  2 = Find cryptocurrencies most similar to a given one
  3 = Naive Bayesian Classification on cryptocurrencies
  4 = Quit


============================================================

Our User Interface for Python:
***************************************************************
*            Cryptocurrency Analysis Application                *
*                                                               *
*                                                               *
*                                                               *
*     1: Compute cosine-similarity between all possible pairs   *
*     2: Categorize and Visualize K-Mean Cluster                *
*     3: Categorize and Visualize Hierarchial Cluster           *
*     4: The Best Competitor in K-Mean Cluster                  *
*     5: The Best Competitor in Hierarchial Cluster             *
*     6: Exit                                                   *
***************************************************************
Enter your choice (1 ~ 6):


# Setup
We will be using Python 2/3 and Python virtual environments:

The path on the top of the /venv/bin/pip should be updated to your p
ath.

$ python -m venv venv

$ . ./venv/bin/activate

$ pip install requests BeautifulSoup4

$ sudo python apt-get python-tk

$ pip install -U gensim

$ pip install -U nltk

$ pip install -U numpy

$ pip install -U scikit-learn

$ python -m pip install -U matplotlib

The twython library should be installed as well

Please download nltk packages as you can get messages when running the code

# To run the Perl program
$ perl vector1.prl

# To run the Python program
$ python similarity.py

# Tools
1. web_robot.py: use Python BeautifulSoup to build a robot that crawls https://coinmarketcap.com/ and the official websites of the first 100 cryptocurrencies on CoinMarketCap; write the website contents to myfile.txt.

2. tokenizer.py: use nltk.tokenize package to tokenize the content of myfile.txt; write to myfile.raw.

3. stemmer.py: use nltk.tokenize package to tokenize the content of myfile.txt first and then use nltk Porter Stemmers to stem each token; write to myfile.stemmed.

4. similarity.py: Outputs pairwise cosine similarity in a matrix form and uses (k-mean clustering and agglomerative clustering to separate these data into different clusters

# To generate new data from online if you want
1. python3 web_robot.py

2. python3 tokenizer.py

3. python3 stemmer.py

4. cat myfile.tokenized | perl make_hist.prl>myfile.tokenized.hist

5. cat myfile.stemmed | perl make_hist.prl>myfile.stemmed.hist





