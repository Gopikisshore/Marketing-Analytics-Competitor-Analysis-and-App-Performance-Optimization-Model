# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:04:18 2022

@author: Gopikishore
"""
#Required packages
import pandas as pd
import nltk
import re
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from autocorrect import Speller
from nltk.stem import WordNetLemmatizer
import itertools
from datetime import datetime

def CustomizedstopwordsAndCreateWordCloud(tokenCategory, stopwordsList, NGram, NofFeatures):
    featuresIdentified = {}
    few_extra_stop_words = ["app","apps","android", "google", "play",'give','nan','fix','href', 'url', 'www', 'sa', 'usg', 'aovvaw', 'game', 'nice', 'tica', 'boca', 'bug','si yeni', 'quest', 'atualizadas', 'lei abr', 'yeni ndi', 'abr', 'mais', 'nova', 'lei', 'ng', 'vers', 'core', 'lipo', 'para', 'lm', 'si', 'updated', 'minor', 'english', 'language', 'improve', 'added', 'update','improvement', 'improved', 'de', 'performance', 'fixed', 'star','doe', 'work','great','day', 'good','feature', 'thing', 'ha', 'wa','love','kid','daughter','year','installed','install', "store",'educational', 'kunda','although', 'would', 'what', 'educ', 'learning', 'http', 'many','nursery','kids', 'x', "product", 'content', 'student','children', 'kiss', 'hello', 'learn']
    
    data1 = pd.read_csv(r"static\Dataset\apps.csv")
    data2 = pd.read_csv(r"static\Dataset\reviews.csv")
    sh = data1.shape
    print(sh)
    data2.comments = data2.content.astype('str')
    duplicate = data2.duplicated()
    print('No. of duplicates is {}'.format(sum(duplicate)))
    data = data1.drop_duplicates()
    print('data shape is: {}'.format(data.shape))
          
    if tokenCategory == "DescriptionAndChanges":
        # Converting the 'description' column to string
        data['recentChanges'] = data['recentChanges'].astype('str')
        data['description'] = data['description'].astype('str')
        des_string1 = ' '.join(data['recentChanges'])
        des_string2 = ' '.join(data['description'])
        des_string = des_string1 + des_string2  
    elif tokenCategory == "content":
        # Converting the 'description' column to string
        data2['content'] = data2['content'].astype('str')
        des_string = ' '.join(data2['content'])
    else:
        print("Error in tokenCategory")
        assert False
    
    # Removing unwanted symbols incase if exists
    des_string = re.sub("[^A-Za-z" "]+", " ", des_string).lower()
    des_string = re.sub("[0-9" "]+"," ", des_string)
    
    # Tokenize the sentence
    word_token = word_tokenize(des_string)
    
    # Spell_check
    now = datetime.now()
    print('Starting spell check:',now)
    spell = Speller(lang = 'en')
    corrected_string = ' '.join(spell(word) for word in word_token)
    now = datetime.now()
    print('End of spell check:',now)
    
    # Lemmatization
    #TODO: why do we need to download it every time? Try to optimize this
    #TODO: wordnet is not used anywhere. Do we really need this?
    nltk.download('wordnet') 
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()
    
    lemmatized_string =' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(corrected_string)])
    
    # stopwords removal
    nltk.download('stopwords')
    stop_words1 = stopwords.words('english')
    
    with open(r"static\stopwords_en.txt") as sw:
        stop_words2 = sw.read()
    stop_words2 = stop_words2.split('\n')
    
    # customized stopwords
    stop_words3 = few_extra_stop_words
    
    # combine all stopwords and remove duplicates from the list
    stop_words = [*set(stop_words1+stop_words2+stop_words3+stopwordsList)]
    
    # remove stopwords
    des_withoutStopwords = [word for word in word_tokenize(lemmatized_string) if not word in stop_words]

    final_string = " ".join(des_withoutStopwords)

    from wordcloud import WordCloud
    NGram = int(NGram)
    if NGram == 1:
        #TFIDF 
        from sklearn.feature_extraction.text import TfidfVectorizer
        #vectorizer = TfidfVectorizer(des_withoutStopwords, use_idf=True, ngram_range=(1, 1))
        vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 1))
        bag_of_words = vectorizer.fit_transform(des_withoutStopwords)
        vectorizer.vocabulary_
        
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        words_freq_len = len(words_freq)
        if words_freq_len > 200:
            words_dict = dict(words_freq[:200])
        else:
            words_dict = dict(words_freq)
        print('creating word cloud')
        
        WC_height = 2500
        WC_width = 3500
        WC_max_words = 200
        wordCloud = WordCloud(background_color = 'White', max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=stop_words)
        
        wordCloud.generate_from_frequencies(words_dict)
        wordCloud.to_file("static/wordCloud.png")
    elif NGram == 2:
        # Bigrams

        # nltk_tokens = nltk.word_tokenize(text)  
        bigrams_list = list(nltk.bigrams(des_withoutStopwords))
        #print(bigrams_list) #TODO: remove if not required

        dictionary2 = [' '.join(tup) for tup in bigrams_list]
        #print (dictionary2) #TODO: remove if not required
        
        # Using count vectoriser to view the frequency of bigrams
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(ngram_range=(2, 2))
        bag_of_words = vectorizer.fit_transform(dictionary2)
        vectorizer.vocabulary_
        
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        words_freq_len = len(words_freq)
        if words_freq_len > 200:
            words_dict = dict(words_freq[:200])
        else:
            words_dict = dict(words_freq)
        print('creating word cloud')
        WC_height = 2500
        WC_width = 3500
        WC_max_words = 200
        wordCloud = WordCloud(background_color = 'White', max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=stop_words)
        
        wordCloud.generate_from_frequencies(words_dict)
        #plt.figure(4,figsize=(25,35) )
        #plt.title('Most frequently occurring words in educational app comments')
        #plt.imshow(wordCloud, interpolation='bilinear')
        #plt.axis("off")
        #plt.show()
        wordCloud.to_file("static/wordCloud.png")

    elif NGram == 3:
        # Trigrams
        # nltk_tokens = nltk.word_tokenize(text)  
        trigrams_list = list(nltk.trigrams(des_withoutStopwords))
        #print(trigrams_list)

        dictionary3 = [' '.join(tup) for tup in trigrams_list]
        #print (dictionary3)
        # Using count vectoriser to view the frequency of bigrams
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(ngram_range=(3, 3))
        bag_of_words = vectorizer.fit_transform(dictionary3)
        vectorizer.vocabulary_
        
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        words_freq_len = len(words_freq)
        if words_freq_len > 200:
            words_dict = dict(words_freq[:200])
        else:
            words_dict = dict(words_freq)
        print('creating word cloud')
        WC_height = 2500
        WC_width = 3500
        WC_max_words = 200
        wordCloud = WordCloud(background_color = 'White', max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=stop_words)
        
        wordCloud.generate_from_frequencies(words_dict)
        #plt.figure(4,figsize=(25,35) )
        #plt.title('Most frequently occurring words in educational app comments')
        #plt.imshow(wordCloud, interpolation='bilinear')
        #plt.axis("off")
        #plt.show()
        wordCloud.to_file("static/wordCloud.png")
    else:
        print("Error in NGram value")
        assert False
    
    #TODO: words_dict is not available for NGram==1. Compute this or it will cause error.
    featuresIdentified = dict(itertools.islice(words_dict.items(), int(NofFeatures)))
    return featuresIdentified


#tc = "DescriptionAndChanges"
#sL = "neet,jee,exam,main exam"
#stopwordsList = [stopword.strip() for stopword in sL.split(',')]
#print(stopwordsList)
#NG = 2 
#NF = 20
#CustomizedstopwordsAndCreateWordCloud(tc,stopwordsList,NG,NF)
