from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None
    """
    try:
        with closing(get(url, stream=True)) as resp: # closing ensures that any network resources are freed when they go out of scope in that with block: prevents fatal errors and network timeouts
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns true if the response seems to be HTML, false otherwise
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)

def log_error(e):
    """
    It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.
    """
    print(e)

raw_html = simple_get('https://coinmarketcap.com/') # main page
html = BeautifulSoup(raw_html, 'html.parser')
for each_div in html.findAll("a", {"class": "currency-name-container"}):
    new_href = "https://coinmarketcap.com"+each_div['href']
    new_html = BeautifulSoup(simple_get(new_href), 'html.parser') # open the new link
    for element in new_html.findAll("a", text="Website"):
        print(element['href'])
