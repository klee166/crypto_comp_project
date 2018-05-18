from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import re
import wikipedia
import urllib3
urllib3.disable_warnings()

def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None
    """
    try:
        with closing(get(url, stream=True, verify=False)) as resp: # closing ensures that any network resources are freed when they go out of scope in that with block: prevents fatal errors and network timeouts
            if is_good_response(resp, url):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp, url):
    """
    Returns true if the response seems to be HTML, false otherwise
    """
    content_type = resp.headers['Content-Type'].lower()
    if (resp.status_code != 200 and resp.status_code != 503) :
        print('status code not 200\n', url)

    elif (content_type is None) :
        print('content type is None\n', url)

    elif (content_type.find('html') <= -1):
        print('content_type.find(html) is not > -1\n', url)

    return ((resp.status_code == 200 or resp.status_code == 503 or resp.status_code == 301)
            and content_type is not None
            and content_type.find('html') > -1)

def log_error(e):
    """
    It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.
    """
    print(e)

# 100 cryptocurrency data for similarity
output_file=open("myfile.txt", "w")
raw_html = simple_get('https://coinmarketcap.com/') # main page
html = BeautifulSoup(raw_html, 'html.parser')
for each_div in html.findAll("a", {"class": "currency-name-container"}):
    if(each_div.text != "Waltonchain"):
        new_href = "https://coinmarketcap.com"+each_div['href']
        new_html = BeautifulSoup(simple_get(new_href), 'html.parser') # open the cryptocurrency website
        for element in new_html.findAll("a", text="Website"):
            website = element['href']
            if(simple_get(website) != None):     # official website
                output_file.write("\n")
                output_file.write(".I ")
                output_file.write(each_div.text) # cryptocurrency name
                output_file.write(" ")
                print(website)
                web_html = BeautifulSoup(simple_get(website), 'html.parser')
                for para in web_html.findAll("p"): # for each p tag
                    ipath = para.text
                    if(len(re.findall(r'[\u4e00-\u9fff]+', ipath)) == 0): # check Chinese character
                        output_file.write(para.text)
    wiki = "https://en.bitcoinwiki.org/wiki/" + each_div.text     # bitcoin wiki
    if(simple_get(wiki) != None):
        wiki_html = BeautifulSoup(simple_get(wiki), 'html.parser')
        for para in wiki_html.findAll("p"): # for each p tag
            output_file.write(para.text)
            sprint(para.text)

"""
# Training data for Bayesian model
training=open("training.txt", "w")
currency = ["Litecoin", "Bitcoin_Cash", "Monero"]
store = ["Bitcoin", "DigixDAO", "Tether", "NuBits"]
protocol = ["RChain", "ZCash"]
service = ["Simple token", "Waves platform"]
utility = ["Steem", "Basic Attention Token","Binance Coin"]

for cur in currency:
    training.write(".I ")
    training.write("1")
    training.write("\n")
    content = wikipedia.page(cur).content;
    training.write(content)
    training.write("\n")
    print(content)

url=["https://en.bitcoinwiki.org/wiki/Nano", "https://en.wikipedia.org/wiki/Dash_(cryptocurrency)"]
for u in url:
    web = BeautifulSoup(simple_get(u), 'html.parser')
    training.write(".I ")
    training.write("1")
    training.write("\n")
    for para in web.findAll("p"):
        training.write(para.text)
    training.write("\n")


for str in store:
    training.write(".I ")
    training.write("2")
    training.write("\n")
    content = wikipedia.page(str).content;
    training.write(content)
    print(content)
    training.write("\n")

url=["https://en.bitcoinwiki.org/wiki/DigixDAO"]
for u in url:
    web = BeautifulSoup(simple_get(u), 'html.parser')
    training.write(".I ")
    training.write("2")
    training.write("\n")
    for para in web.findAll("p"):
        training.write(para.text)
    training.write("\n")

for pro in currency:
    training.write(".I ")
    training.write("3")
    training.write("\n")
    content = wikipedia.page(pro).content;
    training.write(content)
    training.write("\n")

url=["https://en.bitcoinwiki.org/wiki/Ark", "https://en.bitcoinwiki.org/wiki/Zilliqa"]
for u in url:
    web = BeautifulSoup(simple_get(u), 'html.parser')
    training.write(".I ")
    training.write("3")
    training.write("\n")
    for para in web.findAll("p"):
        training.write(para.text)
    training.write("\n")

for ser in service:
    training.write(".I ")
    training.write("4")
    training.write("\n")
    content = wikipedia.page(ser).content;
    training.write(content)
    training.write('\n')


url=["https://en.bitcoinwiki.org/wiki/Lisk", "https://en.bitcoinwiki.org/wiki/Komodo"]
for u in url:
    web = BeautifulSoup(simple_get(u), 'html.parser')
    training.write(".I ")
    training.write("4")
    training.write("\n")
    for para in web.findAll("p"):
        training.write(para.text)
    training.write("\n")


for ut in utility:
    training.write(".I ")
    training.write("5")
    training.write("\n")
    content = wikipedia.page(ut).content;
    training.write(content)
    training.write("\n")

url=["https://en.bitcoinwiki.org/wiki/TRON", "https://en.bitcoinwiki.org/wiki/Dentacoin"]
for u in url:
    web = BeautifulSoup(simple_get(u), 'html.parser')
    training.write(".I ")
    training.write("5")
    training.write("\n")
    for para in web.findAll("p"):
        training.write(para.text)
    training.write("\n")
"""
