#Importing Packages
from google_play_scraper import Sort, reviews, app, search
 
## Extrating App ID's
search_input = "Online certification apps"
ab = []
result = search(search_input,
                lang = "en",  # defaults to 'en'
                country = "in", # defaults to 'us'
                n_hits= 30 # defaults to 30 (= Google's maximum)
)
print(result)

import pandas as pd
result_df = pd.DataFrame(result)

result_df.columns
result_df.head()


for i in result_df['appId']:
    print(i)


app_packages = result_df['appId'].tolist()


print(app_packages)
len(app_packages)
 
# Scraping App's Entire Information

# Required Packages 
import json
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter
from google_play_scraper import Sort, reviews, app

#app_packages = [ 'com.sanaedutech.biology',

#len(app_packages)

app_infos = []

for ap in tqdm(app_packages):
  info = app(ap, lang='en', country='us')
  del info['comments']
  app_infos.append(info)

def print_json(json_object):
  json_str = json.dumps(
    json_object,
    indent=2,
    sort_keys=True,
    default=str
  )
  print(highlight(json_str, JsonLexer(), TerminalFormatter()))

# Sample App Information

print_json(app_infos[0])


# # Conveting into a DataFrame 

app_infos_df = pd.DataFrame(app_infos)
app_infos_df.head()

#scrapping Reviews 

app_reviews = []

for ap in tqdm(app_packages):
  for score in list(range(1, 6)):
    for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
      rvs, _ = reviews(
        ap,
        lang='en',
        country='us',
        sort=sort_order,
        count= 20,
        filter_score_with=score
      )
      for r in rvs:
        r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
        r['appId'] = ap
      app_reviews.extend(rvs)

print_json(app_reviews[0])

len(app_reviews)

app_reviews_df = pd.DataFrame(app_reviews)

app_reviews_df.head()

#import pandas as pd 
#import seaborn as sns
#import matplotlib.pyplot as plt

#app_infos_df = pd.read_csv(r"C:\Users\madym\EdTechapps.csv")
#app_reviews_df = pd.read_csv(r"C:\Users\madym\EdTech_reviews.csv")


#Data Preprocessing

#Data Reduction
# 
#Removal Duplicates & Null Value
app_infos_df.columns

# Data Reduction (Removing unwanted Column) --- App Information
app_infos_df.drop(['histogram','price','free','currency','sale','saleTime','originalPrice','saleText','inAppProductPrice','developer','developerId','developerEmail','developerWebsite','developerAddress','privacyPolicy','genreId','icon','headerImage','screenshots','video','videoImage','contentRating',
       'contentRatingDescription','updated','version','appId','url','recentChangesHTML','descriptionHTML'],axis =1,inplace = True)
app_infos_df.head()

# Data Reduction (Removing unwanted Column) --- App Reviews
app_reviews_df.drop(['reviewId','userName','userImage','reviewCreatedVersion','sortOrder','replyContent','repliedAt'],axis = 1,inplace = True)

### Identify duplicates records in the data
duplicate = app_infos_df.duplicated()
duplicate
sum(duplicate)

#No Duplicates in that dataset
app_infos_df1 = app_infos_df.drop_duplicates()

# Identify missing values in the App Information
# check for count of NA'sin each column
app_infos_df.isna().sum()

# Identify missing values in the App Reviews
# check for count of NA'sin each column
app_reviews_df.isna().sum()

app_reviews1 = app_reviews_df.dropna()


# # Data Visualization

# Installs - typecasting
value_new = []
for value in app_infos_df.installs:
        values = value.replace(',', '').replace('+', '')
        value_new.append(values)
        value_new
        value_series = pd.Series(value_new)
        app_infos_df['Installs'] = value_series    
        app_infos_df['Installs'].dtype # object, convert to integer

plt.figure(figsize=(15,12))
sns.countplot(app_infos_df['Installs'])
plt.title('Installs',fontdict={'size':20,'weight':'bold'})
plt.plot()
plt.savefig(r'static\installs.png')


# # Released date Vs Installs

app_infos_df['Released'] = pd.to_datetime(app_infos_df['released'])


released_date_install=pd.concat([app_infos_df['Installs'],app_infos_df['Released']],axis=1)

plt.figure(figsize=(15,12))
released_date_plot=released_date_install.set_index('Released').resample('3M').mean()
released_date_plot.plot()
plt.title('Released date Vs Installs',fontdict={'size':20,'weight':'bold'})
plt.plot()


# # Is Ad supported?

plt.pie(app_infos_df['adSupported'].value_counts(),radius=2,autopct='%0.2f%%',explode=[0.2,0.5],colors=['#800080','#ff00ff'],labels=['Ad Supported','Ad not supported'],
        startangle=90)
plt.title('Is Ad supported?',fontdict={'size':20,'weight':'bold'})
plt.plot()


# # Text Pre-Processing 

import re
# Dictionary of English Contractions
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have","I've":"I have"}


# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))


# Function for expanding contractions
def expand_contractions(text,contractions_dict=contractions_dict):
  def replace(match):
    return contractions_dict[match.group(0)]
  return contractions_re.sub(replace, text)


# Expanding Contractions in the reviews
app_reviews1['content']=app_reviews1['content'].apply(lambda x:expand_contractions(x))

app_reviews1.head()

#Expanding Contractions in the Description,Recently changes & Summary
app_infos_df1['description']=app_infos_df1['description'].apply(lambda x:expand_contractions(x))
app_infos_df1.head()

app_infos_df1.recentChanges= app_infos_df1.recentChanges.astype('str')
app_infos_df1.dtypes
#app_infos_df1['recentChanges']=app_infos_df1['recentChanges'].apply(lambda x:expand_contractions(x))


app_infos_df1['recentChanges']=app_infos_df1['recentChanges'].apply(lambda x:expand_contractions(x))


# # 2.Lowercase the reviews 
# Lower casing is a common text preprocessing technique. The idea is to convert the input text into same casing format so that 'text', 'Text' and 'TEXT' are treated the same way.

app_reviews1['cleaned']=app_reviews1['content'].apply(lambda x: x.lower())

app_reviews1.head()

app_infos_df1['description_cleaned']=app_infos_df1['description'].apply(lambda x: x.lower())
app_infos_df1['recentChanges_cleaned']=app_infos_df1['recentChanges'].apply(lambda x: x.lower())


app_infos_df1.head()


# # Remove Punctuations
# Punctuations are the marks in English like commas, hyphens, full stops, etc. 
# These are important for English grammar but not for text analysis.Therefore, they need to be removed:

# For handling string
import string
# For performing mathematical operations
import math
app_reviews1['cleaned']=app_reviews1['cleaned'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

app_reviews1.head()

app_infos_df1['description_cleaned']=app_infos_df1['description_cleaned'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
app_infos_df1['recentChanges_cleaned']=app_infos_df1['recentChanges_cleaned'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))


app_infos_df1.head()


# # Text Normalization
# 1.	Stopwords Removal
# 2.	Lemmatization
# 3.	Create Document Term Matrix
# 
# # 1.StopWord Removal 

import nltk

from nltk.corpus import stopwords
", ".join(stopwords.words('english'))

S2 = ["app","apps","android", "google", "play",'give','nan','fix','href', 'url', 'www', 'sa', 'usg', 'aovvaw', 'game', 'nice', 'tica', 'boca', 'bug','si yeni', 'quest', 'atualizadas', 'lei abr', 'yeni ndi', 'abr', 'mais', 'nova', 'lei', 'ng', 'vers', 'core', 'lipo', 'para', 'lm', 'si', 'updated', 'minor', 'english', 'language', 'improve', 'added', 'update','improvement', 'improved', 'de', 'performance', 'fixed', 'star','doe', 'work','great','day', 'good','feature', 'thing', 'ha', 'wa','love','kid','daughter','year','installed','install', "store",'educational', 'kunda','although', 'would', 'what', 'educ', 'learning', 'http', 'many','nursery','kids', 'x', "product", 'content', 'student','children', 'kiss', 'hello', 'learn']

with open(r"C:\Users\madym\Downloads\360 digi\learning resource\data\stopwords_en.txt") as sw:
        stop_words2 = sw.read()
S3 = stop_words2.split('\n')

S1 = stopwords.words('english')

STOPWORDS = set(S1+S2+S3)
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

app_reviews1['cleaned'] = app_reviews1['cleaned'].apply(lambda text: remove_stopwords(text))
app_reviews1.head()

app_infos_df1['description_cleaned'] = app_infos_df1['description_cleaned'].apply(lambda text: remove_stopwords(text))
app_infos_df1['recentChanges_cleaned'] = app_infos_df1['recentChanges_cleaned'].apply(lambda text: remove_stopwords(text))
app_infos_df1.head()


# # 2.Lemmatization 
# Stemming is a faster process than lemmatization as stemming chops off the word irrespective of the context, whereas the latter is context-dependent. Stemming is a rule-based approach, whereas lemmatization is a canonical dictionary-based approach. Lemmatization has higher accuracy than stemming.

nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


app_reviews1["text_lemmatized"] = app_reviews1['cleaned'].apply(lambda text: lemmatize_words(text))

app_reviews1.head()

app_infos_df1["description_lemmatized"] = app_infos_df1['description_cleaned'].apply(lambda text: lemmatize_words(text))
app_infos_df1["recentChanges_lemmatized"] = app_infos_df1['recentChanges_cleaned'].apply(lambda text: lemmatize_words(text))


app_infos_df1.head()


# # Joining all reviews & Description into single Paragraph

# Joinining all the reviews into single paragraph 
ap_reviews_join = " ".join(app_reviews1["text_lemmatized"])
ap_info1_join = " ".join(app_infos_df1["description_lemmatized"])
ap_info2_join = " ".join(app_infos_df1["recentChanges_lemmatized"])


print(ap_reviews_join)

app_info = ap_info1_join + ap_info2_join

print(app_info)

# # Removal of emoji and emoticons
# From Grammarist.com, emoticon is built from keyboard characters that when put together in a certain way represent a facial expression, an emoji is an actual image.
# :-) is an emoticon
# 😀 is an emoji
# Thanks : https://github.com/NeelShah18/emot/blob/master/emot/emo_unicode.py
EMOTICONS = {
    u": \)":"Happy face or smiley",
    u":\)":"Happy face or smiley",
    u":-\]":"Happy face or smiley",
    u":\]":"Happy face or smiley",
    u":-3":"Happy face smiley",
    u":3":"Happy face smiley",
    u":->":"Happy face smiley",
    u":>":"Happy face smiley",
    u"8-\)":"Happy face smiley",
    u":o\)":"Happy face smiley",
    u":-\}":"Happy face smiley",
    u":\}":"Happy face smiley",
    u":-\)":"Happy face smiley",
    u":c\)":"Happy face smiley",
    u":\^\)":"Happy face smiley",
    u"=\]":"Happy face smiley",
    u"=\)":"Happy face smiley",
    u": D":"Laughing, big grin or laugh with glasses",
    u":D":"Laughing, big grin or laugh with glasses",
    u"8 D":"Laughing, big grin or laugh with glasses",
    u"8D":"Laughing, big grin or laugh with glasses",
    u"X D":"Laughing, big grin or laugh with glasses",
    u"XD":"Laughing, big grin or laugh with glasses",
    u"=D":"Laughing, big grin or laugh with glasses",
    u"=3":"Laughing, big grin or laugh with glasses",
    u"B\^D":"Laughing, big grin or laugh with glasses",
    u":-\)\)":"Very happy",
    u": \(":"Frown, sad, andry or pouting",
    u":-\(":"Frown, sad, andry or pouting",
    u":\(":"Frown, sad, andry or pouting",
    u": c":"Frown, sad, andry or pouting",
    u":c":"Frown, sad, andry or pouting",
    u": <":"Frown, sad, andry or pouting",
    u":<":"Frown, sad, andry or pouting",
    u": \[":"Frown, sad, andry or pouting",
    u":\[":"Frown, sad, andry or pouting",
    u":-\|\|":"Frown, sad, andry or pouting",
    u">:\[":"Frown, sad, andry or pouting",
    u":\{":"Frown, sad, andry or pouting",
    u":@":"Frown, sad, andry or pouting",
    u">:\(":"Frown, sad, andry or pouting",
    u":' \(":"Crying",
    u":'\(":"Crying",
    u":' \)":"Tears of happiness",
    u":'\)":"Tears of happiness",
    u"D ':":"Horror",
    u"D:<":"Disgust",
    u"D:":"Sadness",
    u"D8":"Great dismay",
    u"D;":"Great dismay",
    u"D=":"Great dismay",
    u"DX":"Great dismay",
    u": O":"Surprise",
    u":O":"Surprise",
    u": o":"Surprise",
    u":o":"Surprise",
    u":-0":"Shock",
    u"8 0":"Yawn",
    u">:O":"Yawn",
    u":-\*":"Kiss",
    u":\*":"Kiss",
    u":X":"Kiss",
    u"; \)":"Wink or smirk",
    u";\)":"Wink or smirk",
    u"\*-\)":"Wink or smirk",
    u"\*\)":"Wink or smirk",
    u"; \]":"Wink or smirk",
    u";\]":"Wink or smirk",
    u";\^\)":"Wink or smirk",
    u": ,":"Wink or smirk",
    u";D":"Wink or smirk",
    u": P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"X P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"XP":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u": Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u": /":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":-[.]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":S":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u": \|":"Straight face",
    u":\|":"Straight face",
    u":$":"Embarrassed or blushing",
    u": x":"Sealed lips or wearing braces or tongue-tied",
    u":x":"Sealed lips or wearing braces or tongue-tied",
    u": #":"Sealed lips or wearing braces or tongue-tied",
    u":#":"Sealed lips or wearing braces or tongue-tied",
    u": &":"Sealed lips or wearing braces or tongue-tied",
    u":&":"Sealed lips or wearing braces or tongue-tied",
    u"O: \)":"Angel, saint or innocent",
    u"O:\)":"Angel, saint or innocent",
    u"0: 3":"Angel, saint or innocent",
    u"0:3":"Angel, saint or innocent",
    u"0: \)":"Angel, saint or innocent",
    u"0:\)":"Angel, saint or innocent",
    u": b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"0;\^\)":"Angel, saint or innocent",
    u">: \)":"Evil or devilish",
    u">:\)":"Evil or devilish",
    u"\}: \)":"Evil or devilish",
    u"\}:\)":"Evil or devilish",
    u"3: \)":"Evil or devilish",
    u"3:\)":"Evil or devilish",
    u">;\)":"Evil or devilish",
    u"\|; \)":"Cool",
    u"\| O":"Bored",
    u": J":"Tongue-in-cheek",
    u"# \)":"Party all night",
    u"% \)":"Drunk or confused",
    u"%\)":"Drunk or confused",
    u":-###..":"Being sick",
    u":###..":"Being sick",
    u"<: \|":"Dump",
    u"\(>_<\)":"Troubled",
    u"\(>_<\)>":"Troubled",
    u"\(';'\)":"Baby",
    u"\(\^\^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(\^_\^;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(-_-;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(~_~;\) \(・\.・;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(-_-\)zzz":"Sleeping",
    u"\(\^_-\)":"Wink",
    u"\(\(\+_\+\)\)":"Confused",
    u"\(\+o\+\)":"Confused",
    u"\(o\|o\)":"Ultraman",
    u"\^_\^":"Joyful",
    u"\(\^_\^\)/":"Joyful",
    u"\(\^O\^\)／":"Joyful",
    u"\(\^o\^\)／":"Joyful",
    u"\(__\)":"Kowtow as a sign of respect, or dogeza for apology",
    u"_\(\._\.\)_":"Kowtow as a sign of respect, or dogeza for apology",
    u"<\(_ _\)>":"Kowtow as a sign of respect, or dogeza for apology",
    u"<m\(__\)m>":"Kowtow as a sign of respect, or dogeza for apology",
    u"m\(__\)m":"Kowtow as a sign of respect, or dogeza for apology",
    u"m\(_ _\)m":"Kowtow as a sign of respect, or dogeza for apology",
    u"\('_'\)":"Sad or Crying",
    u"\(/_;\)":"Sad or Crying",
    u"\(T_T\) \(;_;\)":"Sad or Crying",
    u"\(;_;":"Sad of Crying",
    u"\(;_:\)":"Sad or Crying",
    u"\(;O;\)":"Sad or Crying",
    u"\(:_;\)":"Sad or Crying",
    u"\(ToT\)":"Sad or Crying",
    u";_;":"Sad or Crying",
    u";-;":"Sad or Crying",
    u";n;":"Sad or Crying",
    u";;":"Sad or Crying",
    u"Q\.Q":"Sad or Crying",
    u"T\.T":"Sad or Crying",
    u"QQ":"Sad or Crying",
    u"Q_Q":"Sad or Crying",
    u"\(-\.-\)":"Shame",
    u"\(-_-\)":"Shame",
    u"\(一一\)":"Shame",
    u"\(；一_一\)":"Shame",
    u"\(=_=\)":"Tired",
    u"\(=\^\·\^=\)":"cat",
    u"\(=\^\·\·\^=\)":"cat",
    u"=_\^=	":"cat",
    u"\(\.\.\)":"Looking down",
    u"\(\._\.\)":"Looking down",
    u"\^m\^":"Giggling with hand covering mouth",
    u"\(\・\・?":"Confusion",
    u"\(?_?\)":"Confusion",
    u">\^_\^<":"Normal Laugh",
    u"<\^!\^>":"Normal Laugh",
    u"\^/\^":"Normal Laugh",
    u"\（\*\^_\^\*）" :"Normal Laugh",
    u"\(\^<\^\) \(\^\.\^\)":"Normal Laugh",
    u"\(^\^\)":"Normal Laugh",
    u"\(\^\.\^\)":"Normal Laugh",
    u"\(\^_\^\.\)":"Normal Laugh",
    u"\(\^_\^\)":"Normal Laugh",
    u"\(\^\^\)":"Normal Laugh",
    u"\(\^J\^\)":"Normal Laugh",
    u"\(\*\^\.\^\*\)":"Normal Laugh",
    u"\(\^—\^\）":"Normal Laugh",
    u"\(#\^\.\^#\)":"Normal Laugh",
    u"\（\^—\^\）":"Waving",
    u"\(;_;\)/~~~":"Waving",
    u"\(\^\.\^\)/~~~":"Waving",
    u"\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",
    u"\(T_T\)/~~~":"Waving",
    u"\(ToT\)/~~~":"Waving",
    u"\(\*\^0\^\*\)":"Excited",
    u"\(\*_\*\)":"Amazed",
    u"\(\*_\*;":"Amazed",
    u"\(\+_\+\) \(@_@\)":"Amazed",
    u"\(\*\^\^\)v":"Laughing,Cheerful",
    u"\(\^_\^\)v":"Laughing,Cheerful",
    u"\(\(d[-_-]b\)\)":"Headphones,Listening to music",
    u'\(-"-\)':"Worried",
    u"\(ーー;\)":"Worried",
    u"\(\^0_0\^\)":"Eyeglasses",
    u"\(\＾ｖ\＾\)":"Happy",
    u"\(\＾ｕ\＾\)":"Happy",
    u"\(\^\)o\(\^\)":"Happy",
    u"\(\^O\^\)":"Happy",
    u"\(\^o\^\)":"Happy",
    u"\)\^o\^\(":"Happy",
    u":O o_O":"Surprised",
    u"o_0":"Surprised",
    u"o\.O":"Surpised",
    u"\(o\.o\)":"Surprised",
    u"oO":"Surprised",
    u"\(\*￣m￣\)":"Dissatisfied",
    u"\(‘A`\)":"Snubbed or Deflated"
}


def remove_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)

remove_emoticons(ap_reviews_join)
remove_emoticons(app_info)

# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

df_reviews = remove_emoji(ap_reviews_join)
df_reviews
df_infos = remove_emoji(app_info)
df_infos


# # UNIGRAM WordCloud  - User Reviews
## Remove single quote early since it causes problems with the tokenizer.
tokens = nltk.word_tokenize(df_reviews)
print(tokens)

Final_Reviews = nltk.Text(tokens)
print(Final_Reviews)

# Using count vectoriser to view the frequency of unigram
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizeru = TfidfVectorizer(ngram_range=(1, 1))
bag_of_wordsu = vectorizeru.fit_transform(Final_Reviews)
vectorizeru.vocabulary_

sum_wordsu = bag_of_wordsu.sum(axis=0)
words_frequ = [(word, sum_wordsu[0, idx]) for word, idx in vectorizeru.vocabulary_.items()]
words_frequ = sorted(words_frequ, key = lambda x: x[1], reverse=True)
print(words_frequ[:100])

from wordcloud import WordCloud
wordcloud = WordCloud(width = 1500, height = 1000, 
                background_color ='white', 
                max_words=200,
                min_font_size = 10).generate(str(words_frequ))                               

import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Most frequently occurring unigrams connected by same colour and font size")
plt.axis("off")
plt.show()


# # Customized Unigram - User Review

Customised_stopwords_input_unigram_reviews = (str(input("Enter any text")))

#Convert string input to list 
def Convert(string):
    li = list(string.split(" "))
    return li

Stopwords_list = Convert(Customised_stopwords_input_unigram_reviews)

# Create a set of stopwords
stopwords_wc = set(STOPWORDS)
customised_words = Stopwords_list # If you want to remove any particular word form text which does not contribute much in meaning
new_stopwords = stopwords_wc.union(customised_words)

# Converting list into a string to plot wordcloud
NewString=' '.join(Final_Reviews)
print('##### Important word combinations ####')
print(NewString)

from wordcloud import WordCloud
wordcloud = WordCloud(width = 1500, height = 1000, 
                background_color ='white', 
                max_words=200,
                stopwords=new_stopwords,
                min_font_size = 10).generate(NewString)

plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Most frequently occurring unigrams connected by same colour and font size")
plt.axis("off")
plt.show()
##BIGRAM - USER REVIEWS

# nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(Final_Reviews))
print(bigrams_list)

dict1 = [' '.join(tup) for tup in bigrams_list]
print (dict1)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(2, 2),stop_words=customised_words)
bag_of_words = vectorizer.fit_transform(dict1)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])


from wordcloud import WordCloud
# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 100

wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,stopwords=stopwords,background_color ='white')

wordCloud.generate_from_frequencies(words_dict)
plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# nltk_tokens = nltk.word_tokenize(text)  
trigrams_list = list(nltk.trigrams(Final_Reviews))
print(trigrams_list)

dict2 = [' '.join(tup) for tup in trigrams_list]
print (dict2)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer1 = TfidfVectorizer(ngram_range=(3,3),stop_words=customised_words)
bag_of_words1 = vectorizer1.fit_transform(dict2)
vectorizer1.vocabulary_

sum_words_ = bag_of_words1.sum(axis=0)
words_freq1 = [(word, sum_words_[0, idx]) for word, idx in vectorizer1.vocabulary_.items()]
words_freq1 = sorted(words_freq1, key = lambda x: x[1], reverse=True)
print(words_freq1[:100])

from wordcloud import WordCloud
# Generating wordcloud
words_dict = dict(words_freq1)
WC_height = 1000
WC_width = 1500
WC_max_words = 100
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,background_color ='white')

wordCloud.generate_from_frequencies(words_dict)
plt.figure(4)
plt.title('Most frequently occurring Trigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()


##Unigram for description & recently changes

## Remove single quote early since it causes problems with the tokenizer.
tokens = nltk.word_tokenize(df_infos)
print(tokens)


# In[80]:


Final_infos = nltk.Text(tokens)
print(Final_infos)


# In[81]:


# Using count vectoriser to view the frequency of unigram
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizerq = TfidfVectorizer(ngram_range=(1, 1))
bag_of_wordsq = vectorizerq.fit_transform(Final_infos)
vectorizerq.vocabulary_

sum_wordsq = bag_of_wordsq.sum(axis=0)
words_freqq = [(word, sum_wordsu[0, idx]) for word, idx in vectorizerq.vocabulary_.items()]
words_freqq = sorted(words_freqq, key = lambda x: x[1], reverse=True)
print(words_freqq[:100])


# In[93]:


from wordcloud import WordCloud
wordcloud = WordCloud(width = 1500, height = 1000, 
                background_color ='white', 
                max_words=100,
                min_font_size = 10).generate(str(words_freqq))


# In[94]:


import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Most frequently occurring unigrams connected by same colour and font size")
plt.axis("off")
plt.show()


# # Bigram for app_infos

# In[84]:


# nltk_tokens = nltk.word_tokenize(text)  
bigrams_list_if = list(nltk.bigrams(Final_infos))
print(bigrams_list_if)

dict_if = [' '.join(tup) for tup in bigrams_list_if]
print (dict_if)


# In[85]:


# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_if = TfidfVectorizer(ngram_range=(2, 2))
bag_of_words_if = vectorizer_if.fit_transform(dict_if)
vectorizer_if.vocabulary_

sum_words_if = bag_of_words_if.sum(axis=0)
words_freq_if = [(word, sum_words_if[0, idx]) for word, idx in vectorizer_if.vocabulary_.items()]
words_freq_if = sorted(words_freq_if, key = lambda x: x[1], reverse=True)
print(words_freq_if[:100])


# In[90]:


from wordcloud import WordCloud
# Generating wordcloud
words_dict = dict(words_freq_if)
WC_height = 1000
WC_width = 1500
WC_max_words = 100

wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,stopwords=stopwords,background_color ='white')

wordCloud.generate_from_frequencies(words_dict)
plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# # Trigram for app_infos
#nltk_tokens = nltk.word_tokenize(text)  
trigrams_list_if = list(nltk.trigrams(Final_infos))
print(trigrams_list_if)

dict_if1 = [' '.join(tup) for tup in trigrams_list_if]
print (dict_if1)
# In[87]:


# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_if1 = TfidfVectorizer(ngram_range=(3, 3))
bag_of_words_if1 = vectorizer_if1.fit_transform(dict_if)
vectorizer_if1.vocabulary_

sum_words_if1 = bag_of_words_if1.sum(axis=0)
words_freq_if1 = [(word, sum_words_if1[0, idx]) for word, idx in vectorizer_if1.vocabulary_.items()]
words_freq_if1 = sorted(words_freq_if1, key = lambda x: x[1], reverse=True)
print(words_freq_if1[:100])


# In[91]:


from wordcloud import WordCloud
# Generating wordcloud
words_dict = dict(words_freq_if1)
WC_height = 1000
WC_width = 1500
WC_max_words = 100
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,background_color ='white')

wordCloud.generate_from_frequencies(words_dict)
plt.figure(4)
plt.title('Most frequently occurring Trigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:




