# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:39:35 2019

@author: badat
"""

import wikipedia
import pdb
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import pickle
#%%
classes = []
with open('./data/classes.txt','r') as file:
    line=file.readline()
    while line:
        class_n = line.strip().split('\t')[-1].replace('+',' ')
        classes.append(class_n)
        line=file.readline()
#%%
num_class = len(classes)
#%%
training_data = []
print('Tokenize wiki text')
for idx,name in enumerate(classes):#num_class
#    if idx%500==0:
#        with open(file_name,'wb') as f:
#            pickle.dump(training_data,f)
    print('-'*50)
    print(idx,name)
    try:
        try:
            page=wikipedia.page(name)
        except wikipedia.exceptions.DisambiguationError as ambiguity:
            page=wikipedia.page(ambiguity.options[0])
        summary = page.summary
        content = page.content
        training_data.append([name,summary,content])
    except Exception as e:
        print(e)
#%%
with open('./data/wiki_article.pkl','wb') as f:
    pickle.dump(training_data,f)
    
#%%
with open('./data/wiki_article.pkl','rb') as f:
    training_data_t=pickle.load(f)