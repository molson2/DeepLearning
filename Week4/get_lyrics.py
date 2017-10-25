from bs4 import BeautifulSoup
import random
from time import sleep
import re
import urllib2
url = 'https://www.azlyrics.com/t/taylorswift.html'

# get list of songs
html = urllib2.urlopen(url).read()
soup = BeautifulSoup(html)
links = soup.find_all('a')[31:229]

songs = []
for link in links:
    try:
        songs.append(link.attrs['href'].split('/')[-1])
    except:
        continue

# get lyrics
base_url = 'https://www.azlyrics.com/lyrics/taylorswift/'
lyrics = {}
exceptions = []
for song in songs:
    try:
        soup = BeautifulSoup(urllib2.urlopen(base_url + song).read())
        div = soup.find_all('div', {'class': ''})[1]
        lyrics[song[:-5]] = unicode(div.text)
        print song
    except:
        exceptions.append(song)
        print 'Bad song: ' + song
    sleep(20 * random.random())


# write to disk
with open('swift.csv', 'w') as f:
    for key in lyrics.keys():
        f.write('##' + key.upper() + '##')
        f.write(lyrics[key].encode('utf-8'))
