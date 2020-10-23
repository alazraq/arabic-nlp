
from wordcloud import WordCloud,STOPWORDS, ImageColorGenerator 
import re
import string
import nltk
import unicodedata #defines the character properties for all unicode characters
from textacy import preprocessing #bibliothèque pour preprocess des données textuelle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from greek_stemmer import GreekStemmer
from nltk.stem.isri import ISRIStemmer
import numpy as np
import pandas as pd
import os
import sys
from corpus import *
from matplotlib import pyplot as plt
from nltk.stem.snowball import FrenchStemmer
from nltk.stem import PorterStemmer
import time
import pickle
import arabic_reshaper
from bidi.algorithm import get_display

class WordImportance():
    
    def __init__(self,*documents):
        
        """
        Parameters
        ----------
        *documents : list of text
            all the documents that compose the corpus
            
        self.documents : list of all the documents
        self.dico_tf_corpus : frequency of each word of the corpus (we process all the documents as one document ) (remember that if we want to compute the tf-idf score, we need to build a dictionnary of tf for each documents)
        self.all_corpus : one big text with all the text of the documents
        self.words : set of all words that are in the dictionnary (count just 1 time)
        self.dico_idf : as this is common to all the corpus, we build the idf dico 
        self.dico_final : key are the words of the corpus, values are array of size nb_docs and at each position we have the tf_idf_ij score (word i doc j)
        """
    
        self.documents = documents
        self.dico_tf_corpus = {}
        self.nb_docs = len(self.documents)
        self.all_corpus = '\n'.join(documents)
        self.words = set(word_tokenize(self.all_corpus))
        self.dico_idf = {}
        self.dico_final = {}
    

    def word_freq(self):
        """
        build the dictionnary of the time frequence of each word for the all corpus (different from building this type of dictionnary for each documents)
        the dico is sorted
        """
        
        self.dico_tf_corpus = self.tf(self.all_corpus)
        tmp = {}
        for k in sorted(self.dico_tf_corpus, key = self.dico_tf_corpus.get, reverse = True): #on trie le dictionnaire 
            tmp[k] = self.dico_tf_corpus[k]
        self.dico_tf_corpus = tmp 
    
    def build_idf_dico(self):
        """
        build the self.dico_idf because it is common so we can run it at the beginning
        """
        for word in self.words :
            self.dico_idf[word] = self.idf(word) 
            
        
    
    def tf(self,text):
        """
        

        Parameters
        ----------
        text : str
            It is the text from where whe want to have the frequence of the words in this text

        Returns
        -------
        dico_tf : dic
            return the dictionnary with in key the word of the text, and the value are the frequence

        """
        
        longueur = len(word_tokenize(text))
        dico_tf = {}
        
        for word in word_tokenize(text):
            dico_tf[word] = dico_tf.get(word,0) + 1
        
        total = sum(dico_tf.values(),0.0)   
        dico_tf = {k : v / total for k,v in dico_tf.items()}
        return dico_tf
        
    
    def idf(self,word_target):
        
        nb = 0
        for doc in self.documents :
            if word_target in word_tokenize(doc) :
                nb += 1
    
        return np.log10(self.nb_docs / (nb ) ) 
    
    
    def tf_idf(self,word_target,dico_tf):
        """
        Parameters
        ----------
        word_target : TYPE
            DESCRIPTION.
        dico_tf_doc : TYPE
            DESCRIPTION.

        Returns
        -------
        score : float
        this is the tf_idf score for word i in document j

        """
        
        try : 
            score = dico_tf[word_target] * self.dico_idf[word_target]
        except KeyError: #le mot n'est pas de le dico du document 
            score = 0
        return score
    
    
    
    def tf_idf_corpus(self):
        
        #initialisation dico final 
        for word in self.words : 
            self.dico_final[word] = np.zeros(self.nb_docs)
        
        for i, doc in enumerate(self.documents) :
            tmp = self.tf(doc)
            for word in self.words : 
                self.dico_final[word][i] = self.tf_idf(word,tmp)
        




class PreprocessData():
    
    def __init__(self,text,language):
        """
        Parameters
        ----------
        text : str
            text we want to preprocess on
        language : str 
            language for the preprocessing (french,english...)
        """
 
        self.text = text.lower()
        self.language = language

        
    def remove_punctuation(self):
        
        """
        Remove all the punctuation anf for english remove all accents for english
        """
        arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
        english_punctuations = string.punctuation
        punctuation = arabic_punctuations + english_punctuations
        table = str.maketrans('','',punctuation) #maketrans build a mapping for translate function
        self.text = self.text.translate(table)
        
        remove_dico = {"'" : " ", '’' : " ", "«": "", "»" : "", "”" : "", "“" : "", "–" : " ", "-" : " ","—":" "}
        table  = str.maketrans(remove_dico)
        self.text = self.text.translate(table)
        
        """Cleaning text by removing vocalization marks"""
        if self.language == 'english':
            #normalize to put in the correct form the text in order to detect the category. Mn category is for any accent, é->e, à->a
            self.text = ''.join([c for c in unicodedata.normalize('NFD',self.text) if unicodedata.category(c) != "Mn"])
            
    def remove_stopwords(self):
        """

        Remove stopwords and words that have a len < 3

        """
        try :
            stop_words = set(stopwords.words(self.language))
        
        except BaseException :
            print("Language need to be french, english, german, spanish or italian")
        
        filterStops = lambda w : len(w) > 1 and w not in stop_words
        filtered = filter(filterStops,word_tokenize(self.text))
        
        
        #on reconstruit tout
        self.text = ' '.join(filtered)
 

    def text_cleaning(self):
        """
        Clean the text by removing currency symbol, or number, and replace one or more spacings with a single space      
        """
        
        arabic_diacritics = re.compile("""
                                        | # Tashdid
                                        | # Fatha
                                        | # Tanwin Fath
                                        | # Damma
                                        | # Tanwin Damm
                                        | # Kasra
                                        | # Tanwin Kasr
                                        | # Sukun                                                             ـ     # Tatwil/Kashida
        
                                """, re.VERBOSE)
        pattern = r'\d*\$*\€*\£*'
        self.text = re.sub(pattern,'',self.text) #replace currency and number
        
        if self.language == 'arabic':
            self.text = re.sub(arabic_diacritics, '', self.text)
            self.text = re.sub("[إأآا]", "ا", self.text)
            self.text = re.sub("ى", "ي", self.text)
            self.text = re.sub("ؤ", "ء", self.text)
            self.text = re.sub("ئ", "ء", self.text)
            self.text = re.sub("ة", "ه", self.text)
            self.text = re.sub("گ", "ك", self.text)
        
        if self.language == 'english' : 
            self.text = preprocessing.unpack_contractions(self.text)  # replace English contractions with their unshortened forms -> I'm -> I am
            
        pattern = r'\s+' # replace one or more spacings with a single space, and one or more linebreaks with a single newline.
        self.text = re.sub(pattern,' ',self.text)
     
    def text_stemming(self):
        """
        stem the text
        """
        if self.language == "french" :
            stemmer = FrenchStemmer()
        elif self.language == "english":
            stemmer = PorterStemmer()
        elif self.language == "italian" :
            stemmer = SnowballStemmer(self.language)
        elif self.language == "german":
            stemmer = SnowballStemmer(self.language)
        elif self.language == "spanish":
            stemmer = SnowballStemmer(self.language)
        elif self.language == "dutch":
            stemmer = SnowballStemmer(self.language)
        elif self.language == "portuguese":
            stemmer = SnowballStemmer(self.language)
        elif self.language == "danish":
            stemmer = SnowballStemmer(self.language)
        elif self.language == "greek":
            stemmer = GreekStemmer()
        elif self.language == "arabic":
            stemmer = ISRIStemmer()
        else : 
            print("Language need to be french, english, german,spanish or italian")
            
            
        self.text = ' '.join([stemmer.stem(word) for word in word_tokenize(self.text) ])

        
    #on ne se sert pas de cette fonction pour l'instant
    def remove_outliers(self,boolean = False): #on enlève les mots les moins commun car sinon va créer trop de bruit de les garder
        
        """
        dico_tf : dictionnary where we have in key a word, and the value is the frequence of apparition
        dico_freq : dico sorted
        """
        
                
        longueur = len(word_tokenize(self.text))
    
        for word in word_tokenize(self.text):
            dico[word] = dico.get(word,0) + 1
        
        total = sum(self.dico_tf.values(),0.0)   
        dico = {k : v / total for k,v in dico.items()}
            
        dico_freq = {}
        for k in sorted(self.dico_tf, key = self.dico_tf.get, reverse = True): #on trie le dictionnaire 
            dico_freq[k] = self.dico_tf[k]
        
        if boolean :     
            rare_words =  list(dico_freq.keys())[-50:] #on enlève les 50 derbiers mots
            self.text = ' '.join([word for word in word_tokenize(self.text) if word not in rare_words])
         
    

        
if __name__ == '__main__':
    
    langue = 'arabic' #on précise la langue pour récuperer les documents et faire le preprocess
    
    # #on lance ça si on veut les docs des articles mis dans un folder
    corpus = RetrieveDocuemnts(langue) #on charge tous les documents
    corpus.retrieve()
    corpus_doc = corpus.documents
    
    #sinon on lance ca 
    # df = pd.read_excel("/Users/johnlevy/Desktop/Python Stage/preprocess/articles/text_label.xlsx",header = 0,index_col = None, sep = " ,")
    # corpus_doc = [title for title in df["Titre"]]
  
    
    #on preprocess les données
    start = time.time()
    liste_doc = []
    for doc in corpus_doc : #corpus.documents si on fait depuis tous les articles
        process = PreprocessData(doc,langue) #create the object
        process.remove_punctuation() #remove punctuations
        process.remove_stopwords() #remove stopwords and words witlor len < 3
        process.text_cleaning()
        #process.text_stemming() #stemming
        liste_doc.append(process.text) #on met tous les textes preprocessés dans la liste des docs
    All_text = ""
    for text in liste_doc:
        All_text = All_text+text
    end = time.time() - start
    print('Preprocessing ended in : ',end," sec")
    
    
  
    #maintenant dans le corpus il y a tous les text preprocessés
    
    #on veut le word count de tout le corpus     
    #on fait tf-idf pour chaque mot du corpus de chaque documents    
    
    print("Building the word count frequency and the tf-idf dico...")
    start = time.time()
    wordcount = WordImportance(*liste_doc)
    print(wordcount)
    wordcount.word_freq() #nous donne le term frequency pour chaque mot du corpus
    wordcount.build_idf_dico() #build the idf dico
    wordcount.tf_idf_corpus() #construit le dico tf-idf
    end = time.time() - start
    print(wordcount.dico_tf_corpus)
    print("Building ended in : ",end/60, " min")
    
 
    

    
    #save the dico
    save_dico_freq = open("time_frequency_dico_corpus.pickle","wb")
    pickle.dump(wordcount.dico_tf_corpus,save_dico_freq)
    save_dico_freq.close()
    
    save_dico_tf_idf_dico = open("td_idf_dico.pickle","wb")
    pickle.dump(wordcount.dico_final,save_dico_tf_idf_dico)
    save_dico_tf_idf_dico.close()
    
    save_idf_dico = open("idf_dico.pickle","wb")
    pickle.dump(wordcount.dico_idf,save_idf_dico)
    save_idf_dico.close()    
  

    #construct the wordCloud for the word frequence
    wordcloud_tf = WordCloud(font_path = 'arial', background_color = "white", max_words = 100).generate_from_frequencies(wordcount.dico_tf_corpus)
    
    plt.figure(figsize = (10,10))
    plt.imshow(wordcloud_tf,interpolation = "bilinear")
    plt.axis("off")
    plt.title('wordcloud.')
    plt.savefig('ok.png')
    plt.show
    
    #construct the wordCloud for the tf-idf dico (need to aggregate the dico because tha values are arrays)

    tmp = {key : value.mean() for (key,value) in wordcount.dico_final.items()}
    wordcloud_tf_idf = WordCloud(font_path='arial',background_color = "white", max_words = 100).generate_from_frequencies(tmp)
    
    plt.figure(figsize = (10,10))
    plt.imshow(wordcloud_tf_idf,interpolation = "bilinear")
    plt.axis("off")
    plt.show
    
    
    #if langue == 'arabic':
    data = arabic_reshaper.reshape(All_text)
    data = get_display(data) # add this line
    WordCloud = WordCloud(font_path='arial', background_color='white',
                          mode='RGB', width=2000, height=1000).generate(data)
    plt.title("arabic")
    plt.imshow(WordCloud)
    plt.axis("off")
    plt.savefig('arabic_reforms.png')
    plt.show()
    