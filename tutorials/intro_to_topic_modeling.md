---
layout: page
title: 3. Unsupervised Machine Learning
permalink: /tutorials/intro_to_topic_modeling/
parent: Tutorials
nav_order: 3
---

# Unsupervised Machine Learning: Topic Models

This tutorial covers:

- how to load a corpus into Python as a datafame
- ways of describing document meta-variables
- how to summarize a corpus with unsupervised machine learning (topic modeling)

---

One of the virtues of computational text analysis is that it enables us to process huge amounts of text that would be difficult (or maybe even impossible) for a single human reader. In fact, when we're dealing with so many documents, we might not even know exactly what kind of themes, issues, or ideas are being featured. 

A **topic model** is a type of statistical model that helps identity "topics" that occur in a collection of documents, or corpus. This approach is particularly useful for synthesizing large corpora---e.g. thousands or millions of documents---that would be otherwise difficult to synthesize by hand. Imagine trying to read all of these documents by yourself and then summarizing which topics are common or unique across all of them! Topic models are a great way to make this process more efficient by leveraging statistical properties of natural language to categorize large bodies of text.

The **key intuition** of a topic model is that we expect certain words to appear together in context if they refer to the same topic. 

For example, we might expect that:

> "dictator", "vizier", and "repression" will appear together more often in documents about **autocrats** <br>
> "democrat", "liberty", and "freedom" will appear together more ofen in documents about **democrats**

These words can be construed as "topics" wherein a single document can contain multiple topics in different proportions. For example, 80% of a document may be devoted to the dictators topic and remaining 20% is devoted to the democrats topic. 

This is basically what a topic model does: leverage the statistical properties of texts to identify clusters of words that co-occur (i.e. topics); then use these word clusters to classify documents by topics.

Let's walk through some popular approaches to this type of modeling and see what it can (and can't) tell us about a body of text.


```python
%matplotlib inline

import warnings, logging

warnings.filterwarnings("ignore",category=DeprecationWarning) #ignore warnings
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR) 
#logging is a means of tracking events that happen when some software runs.
```

Now let's load our text data.

Today we will run some topic models on a corpus of Tweets scraped from Twitter using the `rtweet` package in `R`. I did a simple search for the word "autocray" and collected the most recent 5,000 tweets. I then stored the output as a CSV.

First step is loading the Twitter CSV file into Python.


```python
import pandas as pd

file = "~/autocracy_tweets.csv"

df = pd.read_csv(file)
```

Let's take a peek at the data structure using the "head" command


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>status_id</th>
      <th>created_at</th>
      <th>screen_name</th>
      <th>text</th>
      <th>source</th>
      <th>display_text_width</th>
      <th>reply_to_status_id</th>
      <th>reply_to_user_id</th>
      <th>reply_to_screen_name</th>
      <th>...</th>
      <th>statuses_count</th>
      <th>favourites_count</th>
      <th>account_created_at</th>
      <th>verified</th>
      <th>profile_url</th>
      <th>profile_expanded_url</th>
      <th>account_lang</th>
      <th>profile_banner_url</th>
      <th>profile_background_url</th>
      <th>profile_image_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>x1017830923948122112</td>
      <td>x1316432243099918337</td>
      <td>2020-10-14 17:34:39</td>
      <td>fredmiracle2</td>
      <td>Donald Trump’s first term was characterized by theft, lies, corruption, and the incitement of vi...</td>
      <td>Twitter Web App</td>
      <td>205</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2971</td>
      <td>1426</td>
      <td>2018-07-13 17:59:37</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>http://pbs.twimg.com/profile_images/1127335179075252224/gWWT_5Uv_normal.png</td>
    </tr>
    <tr>
      <th>1</th>
      <td>x1083462522</td>
      <td>x1316432017656033281</td>
      <td>2020-10-14 17:33:45</td>
      <td>Volteer95</td>
      <td>@JenkindLance @Lagiacrus96 @dragbug1 @nick_stryker @MattBaume @feliciaday What the fuck are you ...</td>
      <td>Twitter for Android</td>
      <td>280</td>
      <td>x1316426428595007491</td>
      <td>x1281671562211532802</td>
      <td>JenkindLance</td>
      <td>...</td>
      <td>12132</td>
      <td>46837</td>
      <td>2013-01-12 16:36:51</td>
      <td>False</td>
      <td>https://t.co/C5XASi5cMa</td>
      <td>https://www.youtube.com/user/MegaMapler2000</td>
      <td>NaN</td>
      <td>https://pbs.twimg.com/profile_banners/1083462522/1565878899</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>http://pbs.twimg.com/profile_images/1251117275781967873/zhR96ZAy_normal.jpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>x22080510</td>
      <td>x1316431988648181766</td>
      <td>2020-10-14 17:33:38</td>
      <td>AmbitDiva</td>
      <td>Some of these Trumpers want an autocracy and for Trump to "assign" them a Woman and give them a ...</td>
      <td>Twitter Web App</td>
      <td>199</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>24697</td>
      <td>51948</td>
      <td>2009-02-26 23:43:22</td>
      <td>False</td>
      <td>https://t.co/rX6gnLfgAG</td>
      <td>http://snatchingedges.com</td>
      <td>NaN</td>
      <td>https://pbs.twimg.com/profile_banners/22080510/1508380722</td>
      <td>http://abs.twimg.com/images/themes/theme4/bg.gif</td>
      <td>http://pbs.twimg.com/profile_images/1136467511279542273/CQVxiZFU_normal.png</td>
    </tr>
    <tr>
      <th>3</th>
      <td>x35606154</td>
      <td>x1316431817361195009</td>
      <td>2020-10-14 17:32:58</td>
      <td>bmangh</td>
      <td>Last Exit From Autocracy | America survived one Trump term. It wouldn’t survive a second. https:...</td>
      <td>Twitter for iPhone</td>
      <td>113</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>582373</td>
      <td>154</td>
      <td>2009-04-26 23:57:48</td>
      <td>False</td>
      <td>https://t.co/Nv8Q66R2Ou</td>
      <td>http://phollo.me/bmangh</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>http://pbs.twimg.com/profile_images/2881649256/ac3af9f9f33b7891bab2ed8ffae90b51_normal.jpeg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>x213150691</td>
      <td>x1316431810952298499</td>
      <td>2020-10-14 17:32:56</td>
      <td>AndrewRei15</td>
      <td>@CIAspygirl 2/ all over voting booths correctly pointed out that he is dead? He got almost 70% o...</td>
      <td>Twitter Web App</td>
      <td>277</td>
      <td>x1316429870465916928</td>
      <td>x21445143</td>
      <td>CIAspygirl</td>
      <td>...</td>
      <td>19750</td>
      <td>565</td>
      <td>2010-11-08 03:10:27</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>http://pbs.twimg.com/profile_images/850813293979553792/EgP-qtww_normal.jpg</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 90 columns</p>
</div>



### Exploring the data

What are the variables contained in the dataframe?


```python
df.columns
```




    Index(['user_id', 'status_id', 'created_at', 'screen_name', 'text', 'source',
           'display_text_width', 'reply_to_status_id', 'reply_to_user_id',
           'reply_to_screen_name', 'is_quote', 'is_retweet', 'favorite_count',
           'retweet_count', 'quote_count', 'reply_count', 'hashtags', 'symbols',
           'urls_url', 'urls_t.co', 'urls_expanded_url', 'media_url', 'media_t.co',
           'media_expanded_url', 'media_type', 'ext_media_url', 'ext_media_t.co',
           'ext_media_expanded_url', 'ext_media_type', 'mentions_user_id',
           'mentions_screen_name', 'lang', 'quoted_status_id', 'quoted_text',
           'quoted_created_at', 'quoted_source', 'quoted_favorite_count',
           'quoted_retweet_count', 'quoted_user_id', 'quoted_screen_name',
           'quoted_name', 'quoted_followers_count', 'quoted_friends_count',
           'quoted_statuses_count', 'quoted_location', 'quoted_description',
           'quoted_verified', 'retweet_status_id', 'retweet_text',
           'retweet_created_at', 'retweet_source', 'retweet_favorite_count',
           'retweet_retweet_count', 'retweet_user_id', 'retweet_screen_name',
           'retweet_name', 'retweet_followers_count', 'retweet_friends_count',
           'retweet_statuses_count', 'retweet_location', 'retweet_description',
           'retweet_verified', 'place_url', 'place_name', 'place_full_name',
           'place_type', 'country', 'country_code', 'geo_coords', 'coords_coords',
           'bbox_coords', 'status_url', 'name', 'location', 'description', 'url',
           'protected', 'followers_count', 'friends_count', 'listed_count',
           'statuses_count', 'favourites_count', 'account_created_at', 'verified',
           'profile_url', 'profile_expanded_url', 'account_lang',
           'profile_banner_url', 'profile_background_url', 'profile_image_url'],
          dtype='object')



### What's in a tweet?

Before doing any kind of computation analysis, it is a good idea to explore the corpus yourself -- in short, read some of the actual tweets!

Exploring the corpus by hand helps you refine your analysis. The more you understand about the underlying content, the better you will be able to pre-process the corpus (keep the relevant stuff and filter out irrelevant text). To that end, there's no substitute for deep reading.


```python
df.text[0], df.text[10], df.text[20]
```




    ('Donald Trump’s first term was characterized by theft, lies, corruption, and the incitement of violence, @davidfrum argues. A second term could spell the end of American democracy. \n\nhttps://t.co/kt40nNDGkY',
     'Lenin argued that, under conditions of a proletarian state, and because of the backwardness in Russia because of the autocracy, WWI and the Civil War, state capitalism is a necessarily transitional phase towards socialism.\n\nhttps://t.co/FSE56Sdw9U',
     '*The authoritarian populist defines “the people” to exclude anyone who thinks differently. Only his followers count as legitimate citizens.*   We are living in an era of minority rule.\nhttps://t.co/4FDF1sO8fg')



What kinds of things do you notice about these tweets -- and how might that inform what kinds of terms you want to keep or discard?

### Convert documents (tweets) to vectors

Once we have a sense of what's contained in the corpus, we can make better decisions in deciding how to preprocess our corpus.

Let's start by transforming the data into a document-term matrix using the CountVectorizer class. We specifically want to transform the raw text of each tweet, i.e. the column called **'text'**.

Pay attention to how we need to modify the pandas dataframe object in order to use **vectorizer.fit_transform()**. Also note that the following cell may take awhile to run depending on the size of your corpus.


```python
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
        
vectorizer = CountVectorizer(min_df=20, # min_df: discard words appearing in fewer than n documents, percentage or count
                             max_df=.90, # max_df: discard words appearing in more than n documents, percentage or count
                             decode_error='replace',
                             strip_accents='unicode', # noticed some encoding errors when examining the dataframe
                             stop_words='english')

dtm = vectorizer.fit_transform(df.text.astype(str).to_list()) #requires iterable object over raw text documents
vocab = vectorizer.get_feature_names()

dtm = dtm.toarray()
vocab = np.array(vocab)
```


```python
dtm.shape
```




    (5000, 579)



## A Probabilistic Topic Model

Let's first try running a probabilistic topic model, which represents a topic as a *probability mass function over words.*

The sklearn package has a Latent Dirichlet Allocation package which we can use to estimate the probabilistic model.


```python
from sklearn.decomposition import LatentDirichletAllocation

# Run LDA

lda = LatentDirichletAllocation(n_components=10, 
                                max_iter=5,#maximum number of iterations 
                                learning_method='online', #Method used to update 
                                learning_offset=50,
                                random_state=0).fit(dtm)


# Display top words associated with each topic

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        
no_top_words = 15
display_topics(lda, vocab, no_top_words)
```

    Topic 0:
    party state china russia amp elections like dictatorship far tyranny left self step better communism
    Topic 1:
    trump don democracy stop https just people need know want let say long years make
    Topic 2:
    reject https president trump ve india live senator government come foreign free bidenharris2020 disinformation reason
    Topic 3:
    https like trump election look fascism authoritarianism signs america news just warning read key surviving
    Topic 4:
    senmikelee amp want country democratic fascist liberty https trump mike republicans white lee peace people
    Topic 5:
    amp politics human run making regime did constitution society govt life covid autocratic yes point
    Topic 6:
    democracy country people think isn way mean doesn joebiden america government trying wants right vote
    Topic 7:
    https corruption end togo term american man lies democracy trump hide donald violence office davidfrum
    Topic 8:
    vote https democracy gop voting states biden new amp vs corrupt rights trump time united
    Topic 9:
    democracy realdonaldtrump republic trump power republican oligarchy rule ll law biden putin authoritarian party https


# Exercises

1. Re-run the LDA topic model with different values for min_df and max_df. How does this change the size of your dtm? How does it change the topic results?

2. Now re-run the LDA topic model with 5, 15, and 20 topics. Do you see more or fewer distinct categories when you decrease and increase the number of topics in the modl?

3. Try adding different n-gram combinations to your topic models. This is done in `CountVectorizer()` and `TfidfVectorizer()` using the input `ngram_range=()`. For example, if you wanted to include unigrams and bigrams, you would write `CountVectorizer(ngram_range(1,2))`. Does this lead to different results?

***

# Topic model visualization

The Gensim module has a lot of cool ways to visualize the output of topic models. We'll explore some below.

Before we begin, be sure to execute the following line:


```python
!python -m spacy download en
```

Now download or import the following modules:


```python
# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import re
```

### Preprocessing text for gensim

Gensim has its own text preprocessing tools to create the document-term matrix.

Let's begin by writing a simple function to tokenize our text.


```python
def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\n', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)
```


```python
tweets = df.text.values.astype(str).tolist() # recall this is how we convert the raw text data from our pandas dataframe into an iterable object
tweet_words = list(sent_to_words(tweets))
```


```python
tweet_words
```




    [['donald',
      'trump',
      'first',
      'term',
      'was',
      'characterized',
      'by',
      'theft',
      'lies',
      'corruption',
      'and',
      'the',
      'incitement',
      'of',
      'violence',
      'davidfrum',
      'argues',
      'second',
      'term',
      'could',
      'spell',
      'the',
      'end',
      'of',
      'american',
      'democracy',
      'https',
      'co',
      'kt',
      'nndgky'],
     ['jenkindlance',
      'lagiacrus',
      'dragbug',
      'nick_stryker',
      'mattbaume',
      'feliciaday',
      'what',
      'the',
      'fuck',
      'are',
      'you',
      'talking',
      'about',
      'the',
      'roman',
      'republic',
      'literally',
      'had',
      'system',
      'with',
      'voting',
      'blocks',
      'called',
      'tribes',
      'not',
      'direct',
      'popular',
      'vote',
      'election',
      'system',
      'and',
      'they',
      'still',
      'ended',
      'up',
      'turning',
      'into',
      'an',
      'autocracy',
      'and',
      'what',
      'does',
      'that',
      'have',
      'to',
      'do',
      'with',
      'anything',
      'anyway',
      'what',
      'the',
      'fuuuuuuuuuck'],
     ['some',
      'of',
      'these',
      'trumpers',
      'want',
      'an',
      'autocracy',
      'and',
      'for',
      'trump',
      'to',
      'assign',
      'them',
      'woman',
      'and',
      'give',
      'them',
      'job',
      'and',
      'home',
      'they',
      'watch',
      'the',
      'handmaids',
      'tale',
      'in',
      'delight',
      'women',
      'are',
      'just',
      'necessary',
      'accessories',
      'to',
      'them'],
     ['lol',
      'that',
      'dadabee',
      'protest',
      'starter',
      'pack',
      'dey',
      'funny',
      'me',
      'anyway',
      'the',
      'protests',
      'are',
      'absolutely',
      'necessary',
      'to',
      'demonstrate',
      'the',
      'power',
      'of',
      'the',
      'people',
      'and',
      'keep',
      'autocracy',
      'in',
      'check',
      'so',
      'lets',
      'gooo'],
     ...]



### Stop words

Let's create a customized stop words list. But don't reinvent the wheel when you don't have to -- let's first load the already-made NLTK stop words list as our baseline.


```python
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

print(stop_words)
```

    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


When thinking about what to include or exclude from our list of stopwords, you might consider special nouns or phrases that are unique to certain corpora but don't contribute much substantive meaning. For e.g., tweets frequently post links to other sites, so we might consider adding url notation to the stop words list.


```python
stop_words.extend(['http', 'https', 'url',
                   'from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', 'be', 
                   'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 
                   'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 
                   'run', 'need', 'even', 'right', 'line', 'even', 'also']) 

# Feel free to add or delete as many stop words as you think necessary
```


```python
'due', 'process'
'due_process'
'rule_of_law'
'not','fair'
'not_fair'
```

### Building bigram + trigram models and lemmatizing

Let’s form bigrams and trigrams using the **Phrases** model. Phrases automatically detects common phrases – multi-word expressions, or n-grams – from a stream of sentences.


```python
# Build the bigram and trigram models
bigram = gensim.models.Phrases(tweet_words, min_count=5, threshold=100) # higher threshold means fewer phrases
trigram = gensim.models.Phrases(bigram[tweet_words], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
```

Next, ***lemmatize*** each word to its root form, keeping only nouns, adjectives, verbs and adverbs.

We keep only words with the allowed parts of speech tags (allowed_postags) because they are the ones contributing the most to the meaning of the sentences

We use [Spacy](https://spacy.io/) for lemmatization.


```python
# !python3 -m spacy download en  # may need to run this line in terminal, but only once
spacy.load('en')
from spacy.lang.en import English
parser = English()
```

Let's combine our preprocessing steps in a single function


```python
# THIS CAN TAKE SOME TIME TO EXECUTE -- good time for a break
def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out

data_ready = process_words(tweet_words)  # processed Text Data!
```


```python
data_ready[10]
```




    ['argue',
     'condition',
     'proletarian',
     'state',
     'state',
     'capitalism',
     'necessarily',
     'transitional',
     'phase',
     'socialism',
     'co']



# Topic Modeling with Gensim

There are many ways to run a topic model in Python using several different packages. One popular, powerful tool is the Gensim package, an open-source library for unsupervised topic modeling and natural language processing that uses modern statistical machine learning and is implemented in Python and Cython.

Whichever package you use is up to you, but Gensim has a lot of cool visual features associated with the package.

Let's look at some of these visuals below. But first, let's re-run our LDA topic model so it's in Gensim's format.


```python
import pprint

# Create Dictionary
id2word = corpora.Dictionary(data_ready)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=10,
                                           alpha='symmetric',
                                           iterations=25, 
                                           per_word_topics=True)

```


```python
lda_model.print_topics() # the numbers are the probability weights of each word in each topic
```




    [(0,
      '0.000*"ortozcv" + 0.000*"oldugu" + 0.000*"unutmamalıyız" + 0.000*"oyle" + 0.000*"boyle" + 0.000*"knock" + 0.000*"stone" + 0.000*"addiction" + 0.000*"nytopinion" + 0.000*"voorbereiding"'),
     (1,
      '0.105*"autocracy" + 0.059*"people" + 0.028*"way" + 0.027*"power" + 0.025*"country" + 0.018*"long" + 0.017*"political" + 0.016*"come" + 0.016*"bad" + 0.015*"fascism"'),
     (2,
      '0.000*"ortozcv" + 0.000*"oldugu" + 0.000*"unutmamalıyız" + 0.000*"oyle" + 0.000*"boyle" + 0.000*"knock" + 0.000*"stone" + 0.000*"addiction" + 0.000*"nytopinion" + 0.000*"voorbereiding"'),
     (3,
      '0.094*"autocracy" + 0.062*"vote" + 0.024*"law" + 0.023*"system" + 0.022*"election" + 0.020*"give" + 0.016*"democratic" + 0.013*"still" + 0.012*"dictatorship" + 0.012*"life"'),
     (4,
      '0.089*"autocracy" + 0.037*"amp" + 0.029*"must" + 0.026*"time" + 0.022*"take" + 0.018*"well" + 0.017*"happen" + 0.017*"read" + 0.016*"live" + 0.015*"party"'),
     (5,
      '0.074*"hide" + 0.072*"nowhere" + 0.062*"realdonaldtrump" + 0.048*"togo" + 0.037*"great" + 0.033*"fight" + 0.029*"call" + 0.020*"history" + 0.019*"activist" + 0.015*"land"'),
     (6,
      '0.000*"ortozcv" + 0.000*"oldugu" + 0.000*"unutmamalıyız" + 0.000*"oyle" + 0.000*"boyle" + 0.000*"knock" + 0.000*"stone" + 0.000*"addiction" + 0.000*"nytopinion" + 0.000*"voorbereiding"'),
     (7,
      '0.076*"government" + 0.042*"leader" + 0.030*"become" + 0.022*"free" + 0.021*"move" + 0.021*"elect" + 0.021*"place" + 0.020*"care" + 0.017*"issue" + 0.016*"next"'),
     (8,
      '0.209*"democracy" + 0.043*"state" + 0.039*"end" + 0.027*"stop" + 0.019*"world" + 0.019*"oligarchy" + 0.016*"watch" + 0.015*"corruption" + 0.015*"help" + 0.014*"socialism"'),
     (9,
      '0.155*"co" + 0.145*"autocracy" + 0.036*"trump" + 0.027*"rule" + 0.017*"mean" + 0.014*"year" + 0.012*"much" + 0.011*"allow" + 0.011*"already" + 0.010*"hold"')]



# Wordcloud 

The easiest visual we can make is a word cloud -- a plot that shows the top N words in each topic (more frequent words are larger)



```python

#!pip install wordcloud # un-comment if you need to install wordcloud

from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

# create a list of color names
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(5, 2, figsize=(20,20), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
```


![word_cloud](word_cloud.png)


# An Interactive Topic Model Plot

pyLDAvis is designed to help users interpret the topics in a topic model that has been fit to a corpus of text data. The package extracts information from a fitted LDA topic model to inform an interactive web-based visualization.

The visualization is intended to be used within an IPython notebook but can also be saved to a stand-alone HTML file for easy sharing.


```python
#!pip install pyLDAvis #uncomment to install this on your machine
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
vis
```





<link rel="stylesheet" type="text/css" href="https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css">


<div id="ldavis_el581111404315021600805992593121"></div>
<script type="text/javascript">

var ldavis_el581111404315021600805992593121_data = {"mdsDat": {"x": [0.18941631800378553, 0.185018285353917, 0.23276037506013827, 0.17402827598047751, -0.14288106177244653, -0.15246493074298506, -0.14919413363103082, -0.11222770941728537, -0.11222770941728541, -0.11222770941728541], "y": [-0.024034899166729042, -0.012211044479089326, 0.019599557839382716, 0.022265455392025055, -0.3117974465804056, 0.10556319157144256, 0.20466230082490672, -0.0013490384671776049, -0.0013490384671775288, -0.0013490384671775533], "topics": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "cluster": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Freq": [19.2422609627624, 18.43707521518239, 18.078716749184938, 17.431589801976592, 10.94734493073355, 7.511548969670878, 5.239410435401296, 1.0373510157333956, 1.0373510143072495, 1.037350905047309]}, "tinfo": {"Term": ["democracy", "co", "autocracy", "vote", "people", "government", "hide", "nowhere", "amp", "trump", "state", "realdonaldtrump", "end", "way", "must", "power", "leader", "rule", "country", "time", "togo", "law", "system", "election", "take", "stop", "give", "become", "great", "long", "people", "way", "power", "country", "long", "political", "come", "bad", "fascism", "keep", "thing", "order", "corrupt", "man", "support", "voting", "never", "sure", "else", "abuse", "constitution", "legal", "point", "soon", "opposition", "russian", "slide", "sense", "matter", "accept", "autocracy", "vote", "law", "system", "election", "give", "democratic", "still", "dictatorship", "life", "part", "child", "turn", "understand", "fact", "biden", "remember", "covid", "nation", "feel", "work", "freedom", "let", "opinion", "change", "head", "fear", "cover", "office", "back", "believe", "autocracy", "co", "trump", "rule", "mean", "year", "much", "allow", "already", "hold", "communism", "kill", "oppose", "fascist", "really", "hope", "piece", "survive", "report", "lie", "leave", "provide", "look", "truth", "tell", "always", "dictator", "strong", "day", "news", "reason", "autocracy", "amp", "must", "time", "take", "well", "happen", "read", "live", "party", "authoritarian", "rise", "citizen", "today", "capitalism", "start", "deal", "wait", "guy", "certainly", "last", "public", "medium", "sell", "play", "leadership", "common", "politic", "regime", "agree", "exist", "autocracy", "democracy", "state", "end", "stop", "world", "oligarchy", "watch", "corruption", "help", "socialism", "important", "road", "full", "literally", "tactic", "promote", "simple", "probably", "old", "late", "undermine", "lead", "welcome", "campaign", "direct", "massive", "dangerous", "american", "grow", "obvious", "government", "leader", "become", "free", "move", "elect", "place", "care", "issue", "next", "ever", "put", "autocrat", "minority", "difference", "close", "member", "threat", "maybe", "write", "sign", "suffer", "seriously", "repression", "true", "act", "ground", "oppression", "population", "model", "burn", "signal", "hide", "nowhere", "realdonaldtrump", "togo", "great", "fight", "call", "history", "activist", "land", "destroy", "real", "demand", "theocracy", "patriot", "family", "publish", "least", "spyware", "prevent", "grip", "drive", "rely", "tighten", "accelerate", "sit", "fake", "wing", "tool", "tech", "surveillance", "rdanielkeleman", "movement", "oldugu", "ortozcv", "unutmamal\u0131y\u0131z", "oyle", "boyle", "disease", "ckn", "dispermit", "bare", "jgixbnkj", "conservatism", "cpnyj", "lyur", "partyless", "\u0938\u0930", "\u0445\u0430\u0440\u0438\u043d", "\u0441\u0430\u043d\u0430\u043b", "\u0436\u0430\u043c\u0442\u0430\u0438", "\u0434\u0430\u0445\u0438\u043d", "eagerness", "\u0434\u044d\u044d\u0440\u044d\u044d\u0441", "bartender", "sap", "rubber", "pilot", "hea", "practically", "handmaiden", "fortheruleoflaw", "projectlincoln", "partner", "amplified", "harness", "apathetic", "leftiestat", "inhumane", "ryanjreilly", "breadline", "monster", "extraordinary", "jillwinebank", "loan", "trumplashesout", "bankrupt", "douthat", "correctly", "dufference", "anoint", "ineptitude", "scruple", "four_year", "traitortrump", "lost", "prisonplanet", "cekgyiuqhk", "evey", "rovh", "joncoopertweet", "devastate", "mtf", "knock", "stone", "addiction", "nytopinion", "voorbereiding", "bloodline", "fosgvjw", "aajtakflop", "stock", "appreciate", "franklinfoer", "stratocracy", "hhu", "por", "idiocracy", "defended", "opacity", "fraudulent", "lpwl", "crush", "minor", "inconsequential", "mcb", "oldugu", "ortozcv", "unutmamal\u0131y\u0131z", "oyle", "boyle", "disease", "ckn", "dispermit", "bare", "jgixbnkj", "conservatism", "cpnyj", "lyur", "partyless", "\u0938\u0930", "\u0445\u0430\u0440\u0438\u043d", "\u0441\u0430\u043d\u0430\u043b", "\u0436\u0430\u043c\u0442\u0430\u0438", "\u0434\u0430\u0445\u0438\u043d", "eagerness", "\u0434\u044d\u044d\u0440\u044d\u044d\u0441", "bartender", "sap", "rubber", "pilot", "hea", "practically", "handmaiden", "fortheruleoflaw", "projectlincoln", "partner", "amplified", "harness", "apathetic", "leftiestat", "inhumane", "ryanjreilly", "breadline", "monster", "extraordinary", "jillwinebank", "loan", "trumplashesout", "bankrupt", "douthat", "correctly", "dufference", "anoint", "ineptitude", "scruple", "four_year", "traitortrump", "lost", "prisonplanet", "cekgyiuqhk", "evey", "rovh", "joncoopertweet", "devastate", "mtf", "knock", "stone", "addiction", "nytopinion", "voorbereiding", "bloodline", "fosgvjw", "aajtakflop", "stock", "appreciate", "franklinfoer", "stratocracy", "hhu", "por", "idiocracy", "defended", "opacity", "fraudulent", "lpwl", "crush", "minor", "inconsequential", "mcb", "oldugu", "ortozcv", "unutmamal\u0131y\u0131z", "oyle", "boyle", "disease", "ckn", "dispermit", "bare", "jgixbnkj", "conservatism", "cpnyj", "lyur", "partyless", "\u0938\u0930", "\u0445\u0430\u0440\u0438\u043d", "\u0441\u0430\u043d\u0430\u043b", "\u0436\u0430\u043c\u0442\u0430\u0438", "\u0434\u0430\u0445\u0438\u043d", "eagerness", "\u0434\u044d\u044d\u0440\u044d\u044d\u0441", "bartender", "sap", "rubber", "pilot", "hea", "practically", "handmaiden", "fortheruleoflaw", "projectlincoln", "partner", "amplified", "harness", "apathetic", "leftiestat", "inhumane", "ryanjreilly", "breadline", "monster", "extraordinary", "jillwinebank", "loan", "trumplashesout", "bankrupt", "douthat", "correctly", "dufference", "anoint", "ineptitude", "scruple", "four_year", "traitortrump", "lost", "prisonplanet", "cekgyiuqhk", "evey", "rovh", "joncoopertweet", "devastate", "mtf", "knock", "stone", "addiction", "nytopinion", "voorbereiding", "bloodline", "fosgvjw", "aajtakflop", "stock", "appreciate", "franklinfoer", "stratocracy", "hhu", "por", "idiocracy", "defended", "opacity", "fraudulent", "lpwl", "crush", "minor", "inconsequential", "mcb"], "Freq": [1165.0, 1432.0, 4039.0, 585.0, 575.0, 290.0, 198.0, 192.0, 329.0, 330.0, 242.0, 167.0, 220.0, 279.0, 256.0, 267.0, 163.0, 249.0, 247.0, 227.0, 128.0, 227.0, 216.0, 208.0, 197.0, 152.0, 191.0, 115.0, 99.0, 175.0, 574.2226295938118, 278.80846356828897, 266.60201625139905, 246.30778054002886, 174.39687428597753, 170.583544264425, 157.52627803108345, 155.67262468363026, 146.80674389081244, 139.72085765325454, 138.56789053574522, 128.52129376165104, 122.27060627974838, 117.09402649619096, 98.43047524834864, 92.48427579264344, 92.25992677748336, 88.99123675692383, 82.5000225626056, 81.66564640172606, 80.55470473731323, 79.21052089422076, 77.0815176555139, 76.84189579735161, 76.02424116027373, 74.43866231028024, 70.9450520281886, 65.38792162150705, 63.86114915803435, 62.264049411210024, 1028.1152179633523, 585.0277613935189, 226.39047838341304, 216.02811508962336, 207.85443977352787, 190.57676407493057, 152.14638353365862, 122.84416527392285, 109.82861703803844, 109.72828254103361, 107.95393741546384, 107.71649574510579, 107.07035979996188, 106.74879373593808, 101.99268285936887, 101.39357460462523, 89.2834714224047, 89.22849193630918, 86.40018358174997, 83.09893604797725, 80.36273015523176, 78.92542556834822, 78.266705118343, 76.53668162054154, 72.46054998698436, 72.32141107728611, 72.2221099634223, 68.50005655415168, 62.36803105526241, 59.853945614186834, 56.25089765833323, 879.4683236543049, 1431.3182870355804, 330.1723910130917, 248.60708235220216, 157.3719105792051, 125.0990913002452, 108.85756493236, 104.68249610354397, 100.61930218181588, 93.1528200019296, 92.14172949031942, 89.9320798536022, 87.16880914995525, 84.52942693913589, 78.7999941878316, 77.63697775849036, 76.72186712380083, 73.60450048879281, 74.00367098166022, 70.9334012094035, 68.09992530903428, 67.60171787239611, 66.092166599788, 60.548600936790834, 57.55537611897018, 56.14898797165384, 55.98148904624671, 54.931747248422, 53.90118940380842, 53.19475363707182, 52.66219713470201, 1341.5362868374261, 328.8204205177434, 255.47942968667806, 226.83917301363562, 196.4489790846411, 163.19343139469657, 152.87429280469118, 151.84468903376535, 137.849413727791, 136.3632984658318, 114.47539611558024, 110.52690084359331, 108.73620999085401, 106.26772770515942, 109.53004402010625, 101.51637332864637, 90.24257221203077, 88.52947216699026, 75.24456964369853, 74.94497854403299, 73.94370854153584, 73.44260158680964, 72.54001126056416, 70.60877763011922, 69.77650837203998, 67.64852396068129, 67.52729815601603, 66.6068522565337, 65.7108651390322, 64.78268168648246, 64.23790247530474, 789.5032516633702, 1164.4195128922204, 241.80343525727653, 219.74400758885514, 151.86703756216534, 107.12609574476924, 106.72545279081241, 88.21068152205714, 85.96117283790578, 83.76354369315358, 76.0565383403796, 73.18469357981535, 73.0242554412344, 67.36674409054413, 67.14886770477433, 62.39196061446079, 59.76766294747545, 57.440847145728874, 53.62090731258654, 49.11428767039371, 49.1169292265744, 47.427763247209505, 47.3000897651139, 44.338920886297764, 43.607472419889625, 43.22102003053341, 39.30525878691917, 38.57907263321745, 37.939231047193715, 34.60325238588863, 34.01553213525304, 290.1275709453987, 162.29533791321313, 114.78354828637617, 83.33307148777793, 80.45700252007303, 79.64474802084479, 78.57221985539633, 75.11877349883329, 65.50563121562006, 61.732769115037996, 60.139600850859075, 57.780736061221006, 56.37288630095554, 49.88981065535663, 49.61556139075694, 45.91191721423958, 45.205195073670424, 43.502512636231586, 41.56008642613274, 40.928276632659596, 40.32886011406188, 40.030220451064956, 39.079187582021206, 41.86551586796963, 36.64715516512566, 36.76167158217623, 35.14776648425173, 32.10152489344032, 32.108029237541864, 30.80559052617165, 34.991601570932744, 40.69893695621467, 197.69554311688762, 192.0670158948893, 166.2482332789122, 127.6699179515097, 98.52553675236243, 86.88758424271575, 77.90272318997923, 54.44781708432342, 49.94841077666503, 40.183392081976066, 39.960605089078285, 38.7345150908323, 34.88075554287511, 31.48509054998593, 27.571154622579805, 27.346137768805832, 29.364149418336567, 25.68436381503434, 30.253816622939613, 23.343106953712702, 22.410709044596086, 21.425351939669984, 21.048306356873, 20.991031782983928, 21.663461760039525, 19.50062363302115, 19.366566935602375, 18.81795396559158, 18.545034921923765, 16.940072646814816, 20.840920584993402, 27.26427001949433, 20.63502429049594, 0.06437345634832416, 0.06437345634832416, 0.06437345634832416, 0.06430633665484059, 0.06428438686749008, 0.06424005230051083, 0.06422246783389962, 0.06422077790814605, 0.06421486509275313, 0.06421479580194775, 0.06421435696018031, 0.0642141644857209, 0.0642141644857209, 0.06421412599082903, 0.06421412599082903, 0.06421411829185064, 0.06421411829185064, 0.06421411829185064, 0.06421411829185064, 0.06421411829185064, 0.06421411829185064, 0.06421410674338308, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.06421411829185064, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.06424560326392005, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.06421439930456138, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.0642141028938939, 0.06421411444236146, 0.06421411829185064, 0.0642141028938939, 0.06422110126523785, 0.06421411444236146, 0.0642141028938939, 0.06427960965140767, 0.06427129475476138, 0.06426358422791763, 0.06426303760045292, 0.06425953456529176, 0.06425045362029705, 0.06425045362029705, 0.06424951819442434, 0.0642485327251922, 0.0642480130441518, 0.06424708146776828, 0.06424517212113098, 0.06424517212113098, 0.06424423669525828, 0.06424326277449369, 0.06424053348665934, 0.06424050269074583, 0.06424044109891883, 0.06424037950709181, 0.06423875117316524, 0.06423837392322482, 0.06423837392322482, 0.06423837392322482, 0.06437353709909664, 0.06437353709909664, 0.06437353709909664, 0.0643063365664325, 0.06428438677911218, 0.06424005221219387, 0.06422246774560683, 0.06422077781985559, 0.06421486500447081, 0.06421479571366552, 0.06421435687189868, 0.06421416439743953, 0.06421416439743953, 0.06421412590254771, 0.06421412590254771, 0.06421411820356934, 0.06421411820356934, 0.06421411820356934, 0.06421411820356934, 0.06421411820356934, 0.06421411820356934, 0.0642141066551018, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421411820356934, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06424560317559547, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421439921627968, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421410280561261, 0.06421411435408016, 0.06421411820356934, 0.06421410280561261, 0.06422110117694695, 0.06421411435408016, 0.06421410280561261, 0.06427960956303631, 0.06427129466640147, 0.06426358413956833, 0.06426303751210437, 0.06425953447694802, 0.06425045353196579, 0.06425045353196579, 0.06424951810609437, 0.06424853263686357, 0.0642480129558239, 0.06424708137944166, 0.064245172032807, 0.064245172032807, 0.06424423660693557, 0.06424326268617232, 0.06424053339834171, 0.06424050260242825, 0.06424044101060133, 0.0642403794187744, 0.06423875108485008, 0.06423837383491017, 0.06423837383491017, 0.06423837383491017, 0.0643734956734962, 0.0643734956734962, 0.0643734956734962, 0.06430632979330902, 0.06428438000830057, 0.06424004544605184, 0.06422246098131691, 0.06422077105574366, 0.06421485824098165, 0.06421478895018366, 0.06421435010846305, 0.06421415763402417, 0.06421415763402417, 0.0642141191391364, 0.0642141191391364, 0.06421411144015884, 0.06421411144015884, 0.06421411144015884, 0.06421411144015884, 0.06421411144015884, 0.06421411144015884, 0.06421409989169251, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421411144015884, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06424559640886879, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.0642143924528396, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421409604220374, 0.06421410759067007, 0.06421411144015884, 0.06421409604220374, 0.06422109441280095, 0.06421410759067007, 0.06421409604220374, 0.06427960279272789, 0.06427128789696882, 0.06426357737094777, 0.0642630307435414, 0.06425952770875401, 0.06425044676472824, 0.06425044676472824, 0.06424951133895535, 0.06424852586982835, 0.06424800618884341, 0.06424707461255928, 0.06424516526612573, 0.06424516526612573, 0.06424422984035283, 0.06424325591969215, 0.06424052663214902, 0.0642404958362388, 0.06424043424441836, 0.06424037265259792, 0.0642387443188451, 0.06423836706894492, 0.06423836706894492, 0.06423836706894492], "Total": [1165.0, 1432.0, 4039.0, 585.0, 575.0, 290.0, 198.0, 192.0, 329.0, 330.0, 242.0, 167.0, 220.0, 279.0, 256.0, 267.0, 163.0, 249.0, 247.0, 227.0, 128.0, 227.0, 216.0, 208.0, 197.0, 152.0, 191.0, 115.0, 99.0, 175.0, 575.0159087255283, 279.6017971800238, 267.3953600691944, 247.1010248186342, 175.19016210197046, 171.3768404067828, 158.31964649955614, 156.4658669423718, 147.59999338291746, 140.5141809233267, 139.36118293123226, 129.31463099856234, 123.0639645499414, 117.88732518238707, 99.2236791809096, 93.27763509309507, 93.05345286213291, 89.78503942957342, 83.29334507004562, 82.45987082994063, 81.3479850643037, 80.01524145066291, 77.87474989268726, 77.63527755732859, 76.81754808632792, 75.23200395739934, 71.73845420909747, 66.18590070690507, 64.65437213600103, 63.05770901825601, 4039.085868778666, 585.8252454346942, 227.18827669591295, 216.82575477283672, 208.65192516842987, 191.37432270231415, 152.94391393533962, 123.64177268292944, 110.62628810055537, 110.52581585762165, 108.75169490915194, 108.51408112380096, 107.8678164092669, 107.54645326024773, 102.79038175252781, 102.19107966785637, 90.08135896180221, 90.02601566654063, 87.1977992778064, 83.8964145407607, 81.16018260282875, 79.723027466223, 79.06427694367436, 77.33449519752968, 73.25823809584253, 73.11890242080364, 73.01979037959853, 69.29947490082323, 63.165704209735594, 60.65147543394592, 57.04839783043712, 4039.085868778666, 1432.1281390790243, 330.9821427515996, 249.41702510058107, 158.1817654539749, 125.9088585145891, 109.6674255264081, 105.49250679015546, 101.42910720800006, 93.96258867130771, 92.95208742260702, 90.74202931801506, 87.97953098214278, 85.33930363280295, 79.60983632972038, 78.44687958309207, 77.53172998820374, 74.4142638164914, 74.82337007348615, 71.74326544685134, 68.90985199642626, 68.41157041543762, 66.90193977591359, 61.35850575094007, 58.36516187708469, 56.95899701686768, 56.791292409284104, 55.74153948441421, 54.71099214795322, 54.00549827339113, 53.47202660821886, 4039.085868778666, 329.6239851096228, 256.283144484685, 227.64284128394974, 197.2524764212088, 163.9969083025729, 153.67785031563167, 152.64831931125897, 138.65293493998368, 137.16679066178864, 115.27905990295481, 111.33046275727635, 109.53978871135284, 107.07125754496359, 110.36971765786444, 102.31993350594556, 91.04622499223876, 89.3331729415857, 76.04817557492294, 75.74878803535637, 74.74727832491799, 74.2464495427955, 73.343554996032, 71.41255375529447, 70.58026951498901, 68.4520504374465, 68.33085683645176, 67.41033792066163, 66.51492901140509, 65.58623349584252, 65.04172236241055, 4039.085868778666, 1165.2128684245165, 242.59680509228247, 220.53733578840638, 152.66037078064122, 107.91947689583799, 107.51886149174064, 89.00405841358408, 86.75529825451636, 84.55699676772171, 76.8502280854823, 73.97820989187215, 73.81825722351867, 68.16017378381923, 67.94242811272564, 63.18795962012728, 60.56152926042127, 58.23427961256235, 54.41446893550186, 49.90769777820655, 49.91062071198517, 48.22138763308841, 48.093605615583385, 45.13250082820486, 44.40098215613468, 44.01441416335251, 40.09927399204207, 39.37247505834051, 38.73252594523883, 35.39654451683863, 34.809197484848205, 290.9520559184732, 163.11981646290653, 115.60801794256936, 84.15764985028606, 81.28168370782365, 80.46914501316385, 79.3969342881983, 75.94327351270103, 66.33005489575281, 62.55719313935187, 60.96408489495916, 58.60521842324786, 57.198365255023525, 50.714316442186664, 50.44456194777885, 46.73633680657785, 46.02975133035971, 44.32705020295188, 42.384521695032134, 41.752739400949636, 41.1537305856778, 40.85478631233807, 39.904373735341906, 42.752533937437725, 37.471727633553485, 37.58979699909716, 35.981709903972494, 32.92613437024917, 32.93322723202185, 31.631272171769304, 35.965329410977134, 49.247812199858345, 198.5488913837796, 192.92032383740442, 167.10159457020208, 128.52329675959245, 99.37901194333801, 87.74097185812447, 78.75622100963713, 55.301233977297336, 50.80483729046478, 41.037225638432254, 40.8140338201492, 39.58793055762906, 35.7344602163458, 32.338609338241035, 28.424675432246822, 28.19959207434121, 30.313760832872685, 26.537941808540765, 31.286097311914148, 24.19692217471962, 23.264062981763747, 22.278822439324536, 21.901741701503283, 21.844931473545454, 22.56086628847095, 20.355331286841263, 20.222782721573253, 19.671619749169086, 19.398390612877805, 17.793942371935803, 22.03415459728648, 29.394786453270456, 22.835528354603202, 0.9143588240417664, 0.9143588240417664, 0.9143588240417664, 0.9140582381542248, 0.9139602958390878, 0.913869633093214, 0.9136804313229939, 0.9136718293354549, 0.9136432361942268, 0.9136428879320473, 0.9136409009549928, 0.9136399691649271, 0.9136399691649271, 0.913639772969189, 0.913639772969189, 0.9136397397161732, 0.9136397397161732, 0.9136397397161732, 0.9136397397161732, 0.9136397397161732, 0.9136397397161732, 0.9136396981021502, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.9136399823745305, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.9167086552188666, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.9137211527240913, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.9136689715753715, 0.913679734331583, 0.913639686553683, 0.9192421290572759, 0.9137083172811384, 0.913639686553683, 1.1437727198106526, 1.1407777755803958, 1.379119634111952, 18.866496831543063, 1.2570946281586424, 1.0422292197753404, 1.8159704793922091, 1.501731502921838, 7.749472596188577, 1.3096462292399536, 0.9950997582431899, 1.3307244481331308, 1.3307244481331308, 1.3980327713721872, 3.2509280484981495, 1.385139605219727, 1.4150086666536703, 6.046326727325593, 1.1445338842721595, 1.0420556499794824, 1.5887798420826478, 1.5887757681736283, 1.5887757681736283, 0.9143588240417664, 0.9143588240417664, 0.9143588240417664, 0.9140582381542248, 0.9139602958390878, 0.913869633093214, 0.9136804313229939, 0.9136718293354549, 0.9136432361942268, 0.9136428879320473, 0.9136409009549928, 0.9136399691649271, 0.9136399691649271, 0.913639772969189, 0.913639772969189, 0.9136397397161732, 0.9136397397161732, 0.9136397397161732, 0.9136397397161732, 0.9136397397161732, 0.9136397397161732, 0.9136396981021502, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.9136399823745305, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.9167086552188666, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.9137211527240913, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.9136689715753715, 0.913679734331583, 0.913639686553683, 0.9192421290572759, 0.9137083172811384, 0.913639686553683, 1.1437727198106526, 1.1407777755803958, 1.379119634111952, 18.866496831543063, 1.2570946281586424, 1.0422292197753404, 1.8159704793922091, 1.501731502921838, 7.749472596188577, 1.3096462292399536, 0.9950997582431899, 1.3307244481331308, 1.3307244481331308, 1.3980327713721872, 3.2509280484981495, 1.385139605219727, 1.4150086666536703, 6.046326727325593, 1.1445338842721595, 1.0420556499794824, 1.5887798420826478, 1.5887757681736283, 1.5887757681736283, 0.9143588240417664, 0.9143588240417664, 0.9143588240417664, 0.9140582381542248, 0.9139602958390878, 0.913869633093214, 0.9136804313229939, 0.9136718293354549, 0.9136432361942268, 0.9136428879320473, 0.9136409009549928, 0.9136399691649271, 0.9136399691649271, 0.913639772969189, 0.913639772969189, 0.9136397397161732, 0.9136397397161732, 0.9136397397161732, 0.9136397397161732, 0.9136397397161732, 0.9136397397161732, 0.9136396981021502, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.9136399823745305, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.9167086552188666, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.9137211527240913, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.913639686553683, 0.9136689715753715, 0.913679734331583, 0.913639686553683, 0.9192421290572759, 0.9137083172811384, 0.913639686553683, 1.1437727198106526, 1.1407777755803958, 1.379119634111952, 18.866496831543063, 1.2570946281586424, 1.0422292197753404, 1.8159704793922091, 1.501731502921838, 7.749472596188577, 1.3096462292399536, 0.9950997582431899, 1.3307244481331308, 1.3307244481331308, 1.3980327713721872, 3.2509280484981495, 1.385139605219727, 1.4150086666536703, 6.046326727325593, 1.1445338842721595, 1.0420556499794824, 1.5887798420826478, 1.5887757681736283, 1.5887757681736283], "Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10"], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -2.8385000228881836, -3.561000108718872, -3.605799913406372, -3.684999942779541, -4.030200004577637, -4.052299976348877, -4.131999969482422, -4.143799781799316, -4.202400207519531, -4.2519001960754395, -4.260200023651123, -4.3354997634887695, -4.385300159454346, -4.428599834442139, -4.602200031280518, -4.6645002365112305, -4.666900157928467, -4.703000068664551, -4.778800010681152, -4.788899898529053, -4.802599906921387, -4.819399833679199, -4.846700191497803, -4.849800109863281, -4.860499858856201, -4.8815999031066895, -4.929699897766113, -5.011199951171875, -5.034800052642822, -5.060200214385986, -2.2560999393463135, -2.7771999835968018, -3.726599931716919, -3.773400068283081, -3.812000036239624, -3.8987998962402344, -4.124000072479248, -4.337900161743164, -4.449900150299072, -4.450799942016602, -4.467100143432617, -4.469299793243408, -4.475299835205078, -4.478300094604492, -4.523900032043457, -4.529799938201904, -4.6570000648498535, -4.657599925994873, -4.689799785614014, -4.728799819946289, -4.76230001449585, -4.780300140380859, -4.788700103759766, -4.810999870300293, -4.865799903869629, -4.867700099945068, -4.869100093841553, -4.921999931335449, -5.0157999992370605, -5.0569000244140625, -5.11899995803833, -2.369499921798706, -1.8628000020980835, -3.3296000957489014, -3.613300085067749, -4.0706000328063965, -4.300099849700928, -4.4390997886657715, -4.478300094604492, -4.5177998542785645, -4.594900131225586, -4.605899810791016, -4.630099773406982, -4.661300182342529, -4.6921000480651855, -4.76230001449585, -4.777100086212158, -4.789000034332275, -4.83050012588501, -4.825099945068359, -4.867400169372559, -4.908199787139893, -4.915599822998047, -4.9380998611450195, -5.025700092315674, -5.076399803161621, -5.101200103759766, -5.1041998863220215, -5.1230998039245605, -5.142000198364258, -5.155200004577637, -5.165299892425537, -1.9276000261306763, -3.2971999645233154, -3.5495998859405518, -3.6684999465942383, -3.812299966812134, -3.99780011177063, -4.0630998611450195, -4.069900035858154, -4.166600227355957, -4.1774001121521, -4.352399826049805, -4.387499809265137, -4.403800010681152, -4.426799774169922, -4.396500110626221, -4.472499847412109, -4.590199947357178, -4.609399795532227, -4.771999835968018, -4.776000022888184, -4.789400100708008, -4.796199798583984, -4.808599948883057, -4.835599899291992, -4.847400188446045, -4.878399848937988, -4.880199909210205, -4.893899917602539, -4.90749979019165, -4.9217000007629395, -4.930099964141846, -2.421299934387207, -1.5676000118255615, -3.139400005340576, -3.235100030899048, -3.6045000553131104, -3.9535000324249268, -3.9572999477386475, -4.147799968719482, -4.173699855804443, -4.19950008392334, -4.29610013961792, -4.33459997177124, -4.3368000984191895, -4.417399883270264, -4.420599937438965, -4.494100093841553, -4.537099838256836, -4.5767998695373535, -4.645599842071533, -4.733399868011475, -4.73330020904541, -4.7683000564575195, -4.770999908447266, -4.835700035095215, -4.85230016708374, -4.861199855804443, -4.956200122833252, -4.974800109863281, -4.991600036621094, -5.083600044250488, -5.1006999015808105, -2.5806000232696533, -3.1614999771118164, -3.5078001022338867, -3.828000068664551, -3.8631999492645264, -3.873300075531006, -3.886899948120117, -3.93179988861084, -4.06879997253418, -4.1280999183654785, -4.154200077056885, -4.194200038909912, -4.218900203704834, -4.341100215911865, -4.34660005569458, -4.424200057983398, -4.439700126647949, -4.478099822998047, -4.523799896240234, -4.539100170135498, -4.553800106048584, -4.561299800872803, -4.585299968719482, -4.51639986038208, -4.649600028991699, -4.646399974822998, -4.691299915313721, -4.7820000648498535, -4.781799793243408, -4.823200225830078, -4.695799827575684, -4.5447001457214355, -2.6038999557495117, -2.6328001022338867, -2.7771999835968018, -3.0411999225616455, -3.300299882888794, -3.4260001182556152, -3.5352001190185547, -3.893399953842163, -3.9797000885009766, -4.197199821472168, -4.2027997970581055, -4.23390007019043, -4.338699817657471, -4.441100120544434, -4.57390022277832, -4.582099914550781, -4.510900020599365, -4.644800186157227, -4.480999946594238, -4.7403998374938965, -4.781099796295166, -4.826099872589111, -4.843800067901611, -4.84660005569458, -4.815000057220459, -4.920199871063232, -4.92710018157959, -4.9558000564575195, -4.9704999923706055, -5.060999870300293, -4.853700160980225, -4.585100173950195, -4.863699913024902, -9.014200210571289, -9.014200210571289, -9.014200210571289, -9.015199661254883, -9.015600204467773, -9.016200065612793, -9.016500473022461, -9.016500473022461, -9.016599655151367, -9.016599655151367, -9.016599655151367, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.016200065612793, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.016599655151367, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.016500473022461, -9.01669979095459, -9.01669979095459, -9.015600204467773, -9.015800476074219, -9.015899658203125, -9.015899658203125, -9.015899658203125, -9.01609992980957, -9.01609992980957, -9.01609992980957, -9.01609992980957, -9.01609992980957, -9.01609992980957, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016300201416016, -9.016300201416016, -9.016300201416016, -9.016300201416016, -9.014200210571289, -9.014200210571289, -9.014200210571289, -9.015199661254883, -9.015600204467773, -9.016200065612793, -9.016500473022461, -9.016500473022461, -9.016599655151367, -9.016599655151367, -9.016599655151367, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.016200065612793, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.016599655151367, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.016500473022461, -9.01669979095459, -9.01669979095459, -9.015600204467773, -9.015800476074219, -9.015899658203125, -9.015899658203125, -9.015899658203125, -9.01609992980957, -9.01609992980957, -9.01609992980957, -9.01609992980957, -9.01609992980957, -9.01609992980957, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016300201416016, -9.016300201416016, -9.016300201416016, -9.016300201416016, -9.014200210571289, -9.014200210571289, -9.014200210571289, -9.015199661254883, -9.015600204467773, -9.016200065612793, -9.016500473022461, -9.016500473022461, -9.016599655151367, -9.016599655151367, -9.016599655151367, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.016200065612793, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.016599655151367, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.01669979095459, -9.016500473022461, -9.01669979095459, -9.01669979095459, -9.015600204467773, -9.015800476074219, -9.015899658203125, -9.015899658203125, -9.015899658203125, -9.01609992980957, -9.01609992980957, -9.01609992980957, -9.01609992980957, -9.01609992980957, -9.01609992980957, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016200065612793, -9.016300201416016, -9.016300201416016, -9.016300201416016, -9.016300201416016], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.6467, 1.6452, 1.6451, 1.6448, 1.6435, 1.6434, 1.643, 1.643, 1.6427, 1.6424, 1.6424, 1.6419, 1.6416, 1.6413, 1.64, 1.6395, 1.6395, 1.6392, 1.6385, 1.6384, 1.6383, 1.638, 1.6378, 1.6378, 1.6377, 1.6375, 1.6369, 1.6359, 1.6357, 1.6354, 0.2798, 1.6894, 1.6873, 1.6871, 1.687, 1.6866, 1.6856, 1.6843, 1.6836, 1.6836, 1.6834, 1.6834, 1.6834, 1.6834, 1.683, 1.683, 1.6819, 1.6819, 1.6816, 1.6813, 1.6809, 1.6808, 1.6807, 1.6804, 1.6799, 1.6798, 1.6798, 1.6792, 1.6781, 1.6776, 1.6767, 0.1664, 1.7099, 1.708, 1.7072, 1.7053, 1.704, 1.703, 1.7027, 1.7024, 1.7018, 1.7017, 1.7015, 1.7012, 1.7009, 1.7002, 1.7001, 1.6999, 1.6995, 1.6994, 1.6991, 1.6986, 1.6985, 1.6983, 1.6971, 1.6965, 1.6961, 1.6961, 1.6958, 1.6955, 1.6953, 1.6952, 0.6082, 1.7444, 1.7437, 1.7433, 1.7428, 1.742, 1.7416, 1.7416, 1.7411, 1.741, 1.7399, 1.7396, 1.7395, 1.7394, 1.7392, 1.739, 1.738, 1.7378, 1.7363, 1.7362, 1.7361, 1.736, 1.7359, 1.7356, 1.7354, 1.7351, 1.7351, 1.7349, 1.7347, 1.7346, 1.7345, 0.1145, 2.2114, 2.2088, 2.2085, 2.2069, 2.2047, 2.2047, 2.2031, 2.2029, 2.2026, 2.2017, 2.2013, 2.2013, 2.2004, 2.2003, 2.1994, 2.1989, 2.1984, 2.1974, 2.196, 2.196, 2.1955, 2.1954, 2.1943, 2.194, 2.1939, 2.1921, 2.1917, 2.1914, 2.1894, 2.189, 2.5859, 2.5837, 2.5816, 2.5789, 2.5785, 2.5784, 2.5783, 2.5778, 2.5762, 2.5755, 2.5751, 2.5746, 2.5742, 2.5723, 2.5722, 2.5709, 2.5707, 2.57, 2.5691, 2.5688, 2.5685, 2.5683, 2.5678, 2.5678, 2.5665, 2.5665, 2.5653, 2.5634, 2.5634, 2.5623, 2.5613, 2.3981, 2.9447, 2.9445, 2.9438, 2.9423, 2.9403, 2.9392, 2.9381, 2.9334, 2.932, 2.9279, 2.9278, 2.9272, 2.9248, 2.9222, 2.9185, 2.9182, 2.9171, 2.9163, 2.9154, 2.913, 2.9116, 2.9099, 2.9092, 2.9091, 2.9084, 2.9061, 2.9057, 2.9046, 2.904, 2.8998, 2.8933, 2.8737, 2.8476, 1.915, 1.915, 1.915, 1.9143, 1.914, 1.9134, 1.9134, 1.9134, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9104, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9132, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9132, 1.9133, 1.9073, 1.9132, 1.9133, 1.6897, 1.6921, 1.5023, -1.1137, 1.5949, 1.7822, 1.2269, 1.4169, -0.2241, 1.5537, 1.8284, 1.5377, 1.5377, 1.4884, 0.6445, 1.4976, 1.4762, 0.0239, 1.6884, 1.7822, 1.3604, 1.3604, 1.3604, 1.915, 1.915, 1.915, 1.9143, 1.914, 1.9134, 1.9134, 1.9134, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9104, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9132, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9132, 1.9133, 1.9073, 1.9132, 1.9133, 1.6897, 1.6921, 1.5023, -1.1137, 1.5949, 1.7822, 1.2269, 1.4169, -0.2241, 1.5537, 1.8284, 1.5377, 1.5377, 1.4884, 0.6445, 1.4976, 1.4762, 0.0239, 1.6884, 1.7822, 1.3604, 1.3604, 1.3604, 1.915, 1.915, 1.915, 1.9143, 1.914, 1.9134, 1.9134, 1.9134, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9104, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9132, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9133, 1.9132, 1.9133, 1.9073, 1.9132, 1.9133, 1.6897, 1.6921, 1.5023, -1.1137, 1.5949, 1.7822, 1.2269, 1.4169, -0.2241, 1.5537, 1.8284, 1.5377, 1.5377, 1.4884, 0.6445, 1.4976, 1.4762, 0.0239, 1.6884, 1.7822, 1.3604, 1.3604, 1.3604]}, "token.table": {"Topic": [1, 1, 7, 1, 6, 7, 4, 3, 3, 3, 5, 4, 4, 1, 2, 3, 4, 6, 2, 1, 6, 2, 2, 6, 7, 5, 4, 6, 4, 2, 2, 4, 6, 3, 1, 4, 3, 1, 1, 5, 1, 2, 2, 5, 3, 4, 7, 5, 2, 7, 3, 2, 6, 5, 7, 6, 2, 1, 5, 6, 4, 2, 7, 7, 1, 3, 2, 2, 7, 7, 2, 3, 6, 6, 2, 5, 2, 6, 7, 7, 6, 5, 4, 4, 2, 5, 7, 7, 3, 3, 3, 5, 6, 1, 3, 7, 4, 5, 2, 5, 6, 4, 7, 3, 1, 2, 3, 2, 5, 4, 1, 3, 1, 5, 1, 6, 3, 4, 6, 6, 6, 6, 4, 7, 3, 4, 2, 1, 3, 6, 7, 3, 5, 2, 5, 5, 2, 3, 1, 6, 1, 2, 4, 7, 1, 3, 6, 4, 1, 4, 1, 6, 7, 1, 7, 5, 5, 3, 4, 7, 6, 3, 7, 4, 7, 7, 3, 3, 4, 7, 2, 3, 6, 4, 5, 3, 1, 4, 1, 6, 6, 1, 3, 6, 7, 5, 7, 1, 5, 1, 7, 4, 5, 2, 1, 2, 3, 4, 5, 3, 6, 1, 1, 7, 3, 2, 5, 4, 7, 3, 7, 1, 6, 7, 4, 4, 7, 7, 6, 3, 3, 2, 5, 2, 2, 1, 4, 5, 1, 5, 4, 7, 2, 5, 6, 3], "Freq": [0.6658979971148996, 0.9944230954364575, 0.9751398602651369, 0.98322633291435, 0.9843096519220009, 0.9841582547373725, 0.9910616380207337, 0.9953313575993114, 0.9957693879024283, 0.9831633794994024, 0.9810875762072814, 0.9981069790494302, 0.9889046640037527, 0.2545130342353542, 0.217623499117584, 0.3322533968325344, 0.19558881035596287, 0.9790489597092414, 0.9892587042724884, 0.9970225650394192, 0.9947406939986515, 0.9816226595258076, 0.9883445827979541, 0.9731594447545223, 0.990397952060897, 0.9909690701272184, 0.9966501893299163, 0.9875792355389675, 0.9901148512764737, 0.9828246197486159, 0.9952625399535525, 0.995072213323551, 0.9842448754675566, 0.9992122638692444, 0.997981005474535, 0.9951580171569683, 0.9897572238665459, 0.995722265720182, 0.9913543777511765, 0.9912939236022159, 0.9955442320830425, 0.9956785401151766, 0.9886031203431125, 0.9905397093327611, 0.9870045831735146, 0.9885088591830364, 0.979446724201256, 0.9989591014162448, 0.9938283655030649, 0.9800550510705138, 0.9860666595931395, 0.9943387045583044, 0.9911871184799054, 0.9769526828282281, 0.9425991906525869, 0.9941698769995989, 0.9968755372474823, 0.996478169176674, 0.9975635155540197, 0.984186018758089, 0.9839837826463742, 0.992310742123415, 0.9395343984846946, 0.9574606586088626, 0.9959350039985374, 0.9960240637272727, 0.9860340549555527, 0.9893152222813386, 0.9915550073992501, 0.550669744551515, 0.16538967295310542, 0.16538967295310542, 0.49616901885931625, 0.9862442706949934, 0.9909307575338966, 0.9829787144102815, 0.9980440285978365, 0.9967277910600502, 0.9961861973073941, 0.9456645650093614, 0.972716418797426, 0.9887970839455815, 0.9862169530432684, 0.9955891475951839, 0.9846974943036715, 0.9934127654834788, 0.9972354850235924, 0.9764700733833258, 0.9897556177951315, 0.994303411614751, 0.6152089403897917, 0.9867770537662114, 0.9950240521243117, 0.9963407186381616, 0.9918226501700272, 0.974724762156902, 0.9900026015439699, 0.9817549712066295, 0.9947696390271782, 0.9772608935931176, 0.9931350066031911, 0.9933961008536976, 0.9797293319722467, 0.9867964888899566, 0.9873118991799967, 0.98653909218151, 0.9896399272848548, 0.9952425969123901, 0.9861290192460884, 0.9952908682368223, 0.9932064558438064, 0.9865184809448782, 0.9924731078509563, 0.9725861871648791, 0.9898789190215234, 0.9909277802449, 0.9925290664787861, 0.9953158120566887, 0.9776285706396916, 0.9859148955896709, 0.9800427827138515, 0.9842315802360738, 0.04379141066812318, 0.9196196240305867, 0.9939140950632841, 0.9949932544831809, 0.9862634230711456, 0.988679056717071, 0.9813815573314217, 0.9910930604237522, 0.9952295133084056, 0.9540721926661891, 0.9767533426991405, 0.9815452986027832, 0.9818124694462882, 0.9951742281815317, 0.9956746960502516, 0.9888663764036024, 0.9893572743898419, 0.9718723625483958, 0.9975669342584612, 0.9930879706308956, 0.9914936359146466, 0.9850596207066962, 0.9982332510977305, 0.9931417757828358, 0.9950006345741577, 0.9917785874299363, 0.98876722051894, 0.9939128339462624, 0.997801100744486, 0.9716630494349352, 0.7152908146913374, 0.9985214400538136, 0.9505341147904283, 0.9923831116317035, 0.990727954408043, 0.9939839063342895, 0.9832119980083754, 0.9566612390948198, 0.9896729601982376, 0.034019638196376156, 0.9185302313021563, 0.9957528565385839, 0.985148742322532, 0.9934076357976388, 0.9923396861765346, 0.9911724571115038, 0.9922584445468355, 0.9588278542504504, 0.9879957521260226, 0.9889958167792029, 0.9823979102960552, 0.997031695107593, 0.9889152459798526, 0.9983280006630947, 0.9836239380503944, 0.9942229519377204, 0.9820822759192072, 0.9773364759126407, 0.9719653462940413, 0.020305470544392558, 0.040610941088785116, 0.8325242923200948, 0.10152735272196278, 0.9788049303473124, 0.9825435763322129, 0.9897062988429458, 0.9889365574226199, 0.9918171535245754, 0.958892370016877, 0.9968732045166256, 0.9975399301237482, 0.9948094186212032, 0.12904103957885213, 0.38712311873655636, 0.38712311873655636, 0.12904103957885213, 0.9956742488095348, 0.9866968244638893, 0.9790774499270866, 0.9876674681788555, 0.9912564561472494, 0.9530658372791013, 0.9944330052432825, 0.9961916204387166, 0.9811995888572911, 0.9936503893693366, 0.9553813114968839, 0.9937434958571055, 0.9586064655953495, 0.9974082960288125, 0.9926218820910826, 0.9613213950994225, 0.9971760970811822, 0.9899949102165563, 0.9959283898500417, 0.9794626976624892, 0.987411105296061, 0.9970326412674877, 0.9941571955419632, 0.9919548161986124, 0.9746712466596394, 0.994918909516008, 0.9985913112466126, 0.9863028785858482, 0.9962704454502745, 0.988718959208372, 0.9978476634052665, 0.9749071997469034, 0.9939211762411178, 0.9658584418704284, 0.9857050271990354, 0.9914799726399207, 0.9819714966790295, 0.9927816158028008], "Term": ["aajtakflop", "abuse", "accelerate", "accept", "act", "activist", "agree", "allow", "already", "always", "american", "amp", "authoritarian", "autocracy", "autocracy", "autocracy", "autocracy", "autocrat", "back", "bad", "become", "believe", "biden", "burn", "call", "campaign", "capitalism", "care", "certainly", "change", "child", "citizen", "close", "co", "come", "common", "communism", "constitution", "corrupt", "corruption", "country", "cover", "covid", "dangerous", "day", "deal", "demand", "democracy", "democratic", "destroy", "dictator", "dictatorship", "difference", "direct", "drive", "elect", "election", "else", "end", "ever", "exist", "fact", "fake", "family", "fascism", "fascist", "fear", "feel", "fight", "fosgvjw", "fraudulent", "fraudulent", "fraudulent", "free", "freedom", "full", "give", "government", "great", "grip", "ground", "grow", "guy", "happen", "head", "help", "hide", "history", "hold", "hope", "idiocracy", "important", "issue", "keep", "kill", "land", "last", "late", "law", "lead", "leader", "leadership", "least", "leave", "legal", "let", "lie", "life", "literally", "live", "long", "look", "man", "massive", "matter", "maybe", "mean", "medium", "member", "minority", "model", "move", "movement", "movement", "much", "must", "nation", "never", "news", "next", "nowhere", "nytopinion", "obvious", "office", "old", "oligarchy", "opinion", "oppose", "opposition", "oppression", "order", "part", "party", "patriot", "people", "piece", "place", "play", "point", "politic", "political", "population", "por", "power", "prevent", "probably", "promote", "provide", "public", "publish", "put", "rdanielkeleman", "rdanielkeleman", "read", "real", "realdonaldtrump", "really", "reason", "regime", "rely", "remember", "report", "repression", "rise", "road", "rule", "russian", "sell", "sense", "seriously", "sign", "signal", "signal", "signal", "signal", "simple", "sit", "slide", "socialism", "soon", "spyware", "start", "state", "still", "stock", "stock", "stock", "stock", "stop", "strong", "suffer", "support", "sure", "surveillance", "survive", "system", "tactic", "take", "tech", "tell", "theocracy", "thing", "threat", "tighten", "time", "today", "togo", "tool", "true", "trump", "truth", "turn", "undermine", "understand", "vote", "voting", "wait", "watch", "way", "welcome", "well", "wing", "work", "world", "write", "year"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [2, 4, 10, 5, 9, 8, 6, 1, 3, 7]};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el581111404315021600805992593121", ldavis_el581111404315021600805992593121_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
        new LDAvis("#" + "ldavis_el581111404315021600805992593121", ldavis_el581111404315021600805992593121_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js", function(){
         LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el581111404315021600805992593121", ldavis_el581111404315021600805992593121_data);
            })
         });
}
</script>



# What are we looking at here?

Different elements of pyLDAvis:

1. **Topic Circles:** one for each topic; areas are proportional to the relative weight of a given topic in the corpus, based on the total number of tokens in the corpus.


2. **Red Bars:** represent the estimated number of times a given term was generated by a given topic. When a topic circle is selected, we show the red bars for the most relevant terms for the selected topic.


3. **Blue Bars:** represent the overall frequency of each term in the corpus. When no topic circle is selected, we display the blue bars for the most salient terms in the corpus; when a topic circle is selected, the blue bar will still be displayed if a given term is one of the top terms in the corpus.
