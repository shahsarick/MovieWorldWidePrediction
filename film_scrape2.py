# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 22:20:13 2016
@author: Sarick Shah
"""
# -*- coding: utf-8 -*-

import urllib2
from bs4 import BeautifulSoup
import re 
from collections import defaultdict 
from pandas import Series, DataFrame 
import numpy as np 
from bs4 import BeautifulSoup
import re 
import sys
sys.setrecursionlimit(20000)
import unidecode
import unicodedata

#%%
def solver(list_of_variables, mpaaRating): 
    list_to_append = []
    for category in list_of_variables: 
        try: 
            if category == 'Rotten Tomatoes':  
                rating_index = mpaaRating.index(category)+3
                rating = mpaaRating[rating_index:rating_index+2]                
                list_to_append.extend([rating])                
            else:
                category = (mpaaRating[mpaaRating.index(category)+1])
                list_to_append.append(category)
                
        except (ValueError, AttributeError):
            #print "category didn't work"
            list_to_append.append(None)            
    return list_to_append
#%%
response = urllib2.urlopen('http://www.the-numbers.com/movie/budgets/all')
main_doc = response.read()
def txt_link_downloader(html_link):
    
    soup = BeautifulSoup(html_link, 'html.parser')
    list_df = []    
    batch = soup.find_all('td')
    counter = 0
    for index,i in enumerate(xrange(0,len(batch),6)):
        list_df.append(map(lambda x: x.get_text(), batch[i:i+6]))
        
        url_end = BeautifulSoup(batch[i+2].encode('utf-8'),'html.parser').find('a').get('href') 
        url = 'http://www.the-numbers.com' + url_end
        list_df[index].append(url)
        
        response = urllib2.urlopen(url)
        main_doc = response.read()
        soup = BeautifulSoup(main_doc,'html.parser')
    
        mpaaRating = []
        for tr in soup.findAll('tr'): 
            for td in tr.findAll('td'): 
                mpaaRating.append(td.get_text())
        mpaaRating = [unidecode.unidecode(x).strip() for x in mpaaRating]   
        
        list_of_variables = ['Genre:','Running Time:','MPAA Rating:','Production Companies:','Domestic Releases:','Domestic DVD Sales','Domestic Blu-ray Sales','Total Domestic Video Sales','Rotten Tomatoes']
        
        second_page = solver(list_of_variables,mpaaRating)
        list_df[index].extend(second_page)
        
        response = urllib2.urlopen(url)
        main_doc = response.read()
        soup = BeautifulSoup(main_doc,'html.parser')
        soup = soup.find(text = re.compile('Weekend Box Office Performance')).parent.parent.find('div', attrs = {"id": "box_office_chart"})
        try:
            soup = soup.get_text()
            soup = unicodedata.normalize('NFKD', soup).encode('utf-8').split()[4:35]
            soup.insert(3,'None')
            list_df[index].extend(soup)
        except:
            pass
        
        counter += 1
        #sets upper limit, max is 5230 as of 10/9/2016
        if counter == 2000:
            return DataFrame(list_df)            
import copy
list_df = txt_link_downloader(main_doc)
#%%
df_deepcopy = copy.deepcopy(list_df)
df_deepcopy.to_csv('sarickmovies3.csv', encoding= 'utf-8' )
