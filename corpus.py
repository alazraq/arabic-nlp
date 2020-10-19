# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:23:47 2020

@author: badre
"""

import os
import os.path 
import codecs
import re 
    

class RetrieveDocuemnts :
    
    def __init__(self,language):
        self.nb_docs = 0
        self.documents = []
        self.language = language
        self.path = str(os.getcwd()) + '/articles/' + self.language + '/'
        
    def retrieve(self):
    
        for root, dirs, files in os.walk(self.path):
            for file in files : 
                if file[:3] == "art" : #pour être sur qu'on ne prend que les fichiers articles
                    with open(self.path + file, "r", encoding = 'utf-8',errors = 'ignore') as f :
                        self.documents.append(f.read())
                      
                        
        self.nb_docs = len(self.documents)
                    
            
        
if __name__ == "__main__":
    
    language = "arabic" #need to specify in which field are the corpus (french field of english filed)
    
    corpus = RetrieveDocuemnts(language)
    corpus.retrieve()