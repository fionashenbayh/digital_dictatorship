---
layout: page
title: 5. Sentiment Analysis
permalink: /tutorials/sentiment-analysis/
parent: Tutorials
nav_order: 5
---

# Sentiment Analysis

One of the most fascinating applications of natural language processing is sentiment analysis: evaluating the opinions, attitudes, or emotive content of texts. Also known as opinion mining or emotion Aritificial Intelligence (AI), this type of computational analysis has a variety of applications in [social science research](https://hedonometer.org/about.html) as well as [private industry](https://www.wired.com/2014/06/everything-you-need-to-know-about-facebooks-manipulative-experiment/). 

Howver, sentiment analysis is among the most challenging types of natural language processing because of the way that sentiment is expressed in the written word. That is, the complexities of human language enable us to express extremely complex ideas and emotions in ways that are not always easy to classify with a simple algorithm. 

To illustrate why this is the case, let's start with some simple examples:

- This welfare policy has wide public support.
- This industry was the best economic performer in 2020.
- I hate feeling tired.

Each of these sentences conveys a single sentiment as indicated by terms like "public support", "best", and "hate". We thus might write a simple classification algorithm that instructs our computer to label sentences as **positive** or **negative** based on whether such terms are used.

But what about more complex sentences?

- President Otto Krat lacks popular support.
- She wasn't opposed to the plan, but she wasn't a fan of it, either.
- Great idea, genius.

The first sentence uses the phrase "popular support" but it's _negated_, meaning if we were just instructing our computer to look for instances of positive phrases, this sentence might be miscategorized as positive. The second sentence is even trickier because there are lots of positive and negative terms being used ("opposed", "fan") but the overarching sentiment is kind of in the middle -- it's neither positive nor negative. The final sentence is the trickiest of all, because it seems to be sarcastic. Whether this sentence is genuine or tongue-in-cheek will depend a lot on subtext, which is something a human reader can pick up on more easily than a computer can.

Grappling with the nuances of human language in a computational environment is a perrennial challenge of sentiment analysis. But the only way to improve is to test things out and iteratively improve. So to that end, let's get started with some existing sentiment analysis packages that show the incredible power of these tools as well as their limitations.

---

### TextBlob

TextBlob is a Python library for processing text data. You can treat TextBlob objects the same way you treat string objects in Python. The advantage of using Textblob is that it has many pre-built functions for parts-of-speech tagging, noun phrase extraction, and **sentiment analysis**.


```python
from textblob import TextBlob

text = '''
The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
'''

blob = TextBlob(text) # create a textblob object and name it 'blob'
```

### TextBlobs Are Like Python Strings

We can treat TextBlob objects just as we would any raw string object.


```python
blob[0:50] # we can slice them
```




    TextBlob("
    The titular threat of The Blob has always struck ")




```python
blob.upper(), blob.lower() # we can convert them to upper or lower case
```




    (TextBlob("
     THE TITULAR THREAT OF THE BLOB HAS ALWAYS STRUCK ME AS THE ULTIMATE MOVIE
     MONSTER: AN INSATIABLY HUNGRY, AMOEBA-LIKE MASS ABLE TO PENETRATE
     VIRTUALLY ANY SAFEGUARD, CAPABLE OF--AS A DOOMED DOCTOR CHILLINGLY
     DESCRIBES IT--"ASSIMILATING FLESH ON CONTACT.
     SNIDE COMPARISONS TO GELATIN BE DAMNED, IT'S A CONCEPT WITH THE MOST
     DEVASTATING OF POTENTIAL CONSEQUENCES, NOT UNLIKE THE GREY GOO SCENARIO
     PROPOSED BY TECHNOLOGICAL THEORISTS FEARFUL OF
     ARTIFICIAL INTELLIGENCE RUN RAMPANT.
     "), TextBlob("
     the titular threat of the blob has always struck me as the ultimate movie
     monster: an insatiably hungry, amoeba-like mass able to penetrate
     virtually any safeguard, capable of--as a doomed doctor chillingly
     describes it--"assimilating flesh on contact.
     snide comparisons to gelatin be damned, it's a concept with the most
     devastating of potential consequences, not unlike the grey goo scenario
     proposed by technological theorists fearful of
     artificial intelligence run rampant.
     "))




```python
blob.find("doomed") # we can locate particular terms
```




    183




```python
apple_blob = TextBlob('apples')  
apple_blob == 'apples' # we can make direct comparisons between TextBlobs and strings
```




    True



### N-grams

 TextBlob can generate any number of n-grams. The most common form is the bigram.


```python
blob.ngrams(n=2)
```




    [WordList(['The', 'titular']),
     WordList(['titular', 'threat']),
     WordList(['threat', 'of']),
     WordList(['of', 'The']),
     WordList(['The', 'Blob']),
     WordList(['Blob', 'has']),
     WordList(['has', 'always']),
     WordList(['always', 'struck']),
     WordList(['struck', 'me']),
     WordList(['me', 'as']),
     WordList(['as', 'the']),
     WordList(['the', 'ultimate']),
     WordList(['ultimate', 'movie']),
     WordList(['movie', 'monster']),
     WordList(['monster', 'an']),
     WordList(['an', 'insatiably']),
     WordList(['insatiably', 'hungry']),
     WordList(['hungry', 'amoeba-like']),
     WordList(['amoeba-like', 'mass']),
     WordList(['mass', 'able']),
     WordList(['able', 'to']),
     WordList(['to', 'penetrate']),
     WordList(['penetrate', 'virtually']),
     WordList(['virtually', 'any']),
     WordList(['any', 'safeguard']),
     WordList(['safeguard', 'capable']),
     WordList(['capable', 'of']),
     WordList(['of', 'as']),
     WordList(['as', 'a']),
     WordList(['a', 'doomed']),
     WordList(['doomed', 'doctor']),
     WordList(['doctor', 'chillingly']),
     WordList(['chillingly', 'describes']),
     WordList(['describes', 'it']),
     WordList(['it', 'assimilating']),
     WordList(['assimilating', 'flesh']),
     WordList(['flesh', 'on']),
     WordList(['on', 'contact']),
     WordList(['contact', 'Snide']),
     WordList(['Snide', 'comparisons']),
     WordList(['comparisons', 'to']),
     WordList(['to', 'gelatin']),
     WordList(['gelatin', 'be']),
     WordList(['be', 'damned']),
     WordList(['damned', 'it']),
     WordList(['it', "'s"]),
     WordList(["'s", 'a']),
     WordList(['a', 'concept']),
     WordList(['concept', 'with']),
     WordList(['with', 'the']),
     WordList(['the', 'most']),
     WordList(['most', 'devastating']),
     WordList(['devastating', 'of']),
     WordList(['of', 'potential']),
     WordList(['potential', 'consequences']),
     WordList(['consequences', 'not']),
     WordList(['not', 'unlike']),
     WordList(['unlike', 'the']),
     WordList(['the', 'grey']),
     WordList(['grey', 'goo']),
     WordList(['goo', 'scenario']),
     WordList(['scenario', 'proposed']),
     WordList(['proposed', 'by']),
     WordList(['by', 'technological']),
     WordList(['technological', 'theorists']),
     WordList(['theorists', 'fearful']),
     WordList(['fearful', 'of']),
     WordList(['of', 'artificial']),
     WordList(['artificial', 'intelligence']),
     WordList(['intelligence', 'run']),
     WordList(['run', 'rampant'])]



### Tokenization

TextBlob has convenient word and sentence tokenizers.


```python
zen = TextBlob("Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex.")
zen.words, zen.sentences
```




    (WordList(['Beautiful', 'is', 'better', 'than', 'ugly', 'Explicit', 'is', 'better', 'than', 'implicit', 'Simple', 'is', 'better', 'than', 'complex']),
     [Sentence("Beautiful is better than ugly."),
      Sentence("Explicit is better than implicit."),
      Sentence("Simple is better than complex.")])



### Parts-of-speech

 A TextBlob object can also classify the part-of-speech (POS) of each word. Tags include DT-determiner, JJ-adjective, NN-noun, etc. For a more detailed explainer on POS tags, see [https://www.clips.uantwerpen.be/pages/mbsp-tags](http://)
 
 Why would we be interested in POS tags when doing sentiment analysis?


```python
blob.tags #returns tuple of (word, part-of-speech)
```




    [('The', 'DT'),
     ('titular', 'JJ'),
     ('threat', 'NN'),
     ('of', 'IN'),
     ('The', 'DT'),
     ('Blob', 'NNP'),
     ('has', 'VBZ'),
     ('always', 'RB'),
     ('struck', 'VBN'),
     ('me', 'PRP'),
     ('as', 'IN'),
     ('the', 'DT'),
     ('ultimate', 'JJ'),
     ('movie', 'NN'),
     ('monster', 'NN'),
     ('an', 'DT'),
     ('insatiably', 'RB'),
     ('hungry', 'JJ'),
     ('amoeba-like', 'JJ'),
     ('mass', 'NN'),
     ('able', 'JJ'),
     ('to', 'TO'),
     ('penetrate', 'VB'),
     ('virtually', 'RB'),
     ('any', 'DT'),
     ('safeguard', 'NN'),
     ('capable', 'JJ'),
     ('of', 'IN'),
     ('as', 'IN'),
     ('a', 'DT'),
     ('doomed', 'JJ'),
     ('doctor', 'NN'),
     ('chillingly', 'RB'),
     ('describes', 'VBZ'),
     ('it', 'PRP'),
     ('assimilating', 'VBG'),
     ('flesh', 'NN'),
     ('on', 'IN'),
     ('contact', 'NN'),
     ('Snide', 'JJ'),
     ('comparisons', 'NNS'),
     ('to', 'TO'),
     ('gelatin', 'VB'),
     ('be', 'VB'),
     ('damned', 'VBN'),
     ('it', 'PRP'),
     ("'s", 'VBZ'),
     ('a', 'DT'),
     ('concept', 'NN'),
     ('with', 'IN'),
     ('the', 'DT'),
     ('most', 'RBS'),
     ('devastating', 'JJ'),
     ('of', 'IN'),
     ('potential', 'JJ'),
     ('consequences', 'NNS'),
     ('not', 'RB'),
     ('unlike', 'IN'),
     ('the', 'DT'),
     ('grey', 'NN'),
     ('goo', 'NN'),
     ('scenario', 'NN'),
     ('proposed', 'VBN'),
     ('by', 'IN'),
     ('technological', 'JJ'),
     ('theorists', 'NNS'),
     ('fearful', 'NN'),
     ('of', 'IN'),
     ('artificial', 'JJ'),
     ('intelligence', 'NN'),
     ('run', 'NN'),
     ('rampant', 'NN')]



### Noun phrases

 We can also use Textblob to extract noun phrases, including both proper and regular nouns.


```python
blob.noun_phrases
```




    WordList(['titular threat', 'blob', 'ultimate movie monster', 'amoeba-like mass', 'snide', 'potential consequences', 'grey goo scenario', 'technological theorists fearful', 'artificial intelligence run rampant'])



### Spellcheck

TextBlob can attempt to correct spelling errors. But it will not work with 100% accuracy, especially when dealing with texts from non-Western countries that have many foreign names.


```python
b = TextBlob("I havv verry goood speling!")
print(b.correct())
```

    I have very good spelling!


Word objects have a spellcheck() and Word.spellcheck() method that returns a list of (word, confidence) tuples with spelling suggestions.


```python
from textblob import Word
w = Word('conandrum')
w.spellcheck()
```




    [('conundrums', 1.0)]



### Classifying Sentiment

 TextBlob also has a built-in sentiment analyzer. The sentiment property returns a named tuple of the form Sentiment(polarity, subjectivity). 

* The **polarity score** is a float within the range [-1.0, 1.0] $\rightarrow$ 1.0 is very negative and 1.0 is very positive.

* The **subjectivity score** is a float within the range [0.0, 1.0] $\rightarrow$ 0.0 is very objective and 1.0 is very subjective; subjectivity is a measure of the extent to which something expressed is a subjective opinion or an objective fact.


```python
blob.sentiment
```




    Sentiment(polarity=-0.1590909090909091, subjectivity=0.6931818181818182)




```python
blob.sentiment.polarity
```




    -0.1590909090909091



Above, we calculated the sentiment for the entire TextBlob object. Can we create a function to calculate the sentiment for each word or sentence?


```python
def get_textBlob_score(text):
    # Polarity score is between -1 to 1
    polarity_scores = []
    sents = TextBlob(text).sentences
    for sent in sents:
        polarity = sent.sentiment.polarity
        polarity_scores.append(polarity)
    return polarity_scores

text = '''
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
'''

get_textBlob_score(text)
```




    [-1.0, -0.075, -0.05]



### Supervised Classification

 The textblob.classifiers module makes it simple to create custom classifiers.

Let’s create a custom sentiment analyzer.


```python
train = [
    ('I love this sandwich.', 'pos'),
    ('this is an amazing place!', 'pos'),
    ('I feel very good about these beers.', 'pos'),
    ('this is my best work.', 'pos'),
    ("what an awesome view", 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
    ('he is my sworn enemy!', 'neg'),
    ('my boss is horrible.', 'neg')
 ]

test = [
    ('the beer was good.', 'pos'),
    ('I do not enjoy my job', 'neg'),
    ("I ain't feeling dandy today.", 'neg'),
    ("I feel amazing!", 'pos'),
    ('Gary is a friend of mine.', 'pos'),
    ("I can't believe I'm doing this.", 'neg')
]
```


```python
from textblob.classifiers import NaiveBayesClassifier
cl = NaiveBayesClassifier(train)
```

### Classifying Text

Call the classify(text) method to use the classifier.


```python
cl.classify("This is an amazing library!")
```




    'pos'



You can get the label probability distribution with the prob_classify(text) method.


```python
prob_dist = cl.prob_classify("This one's a doozy.")
prob_dist.max(), round(prob_dist.prob("pos"), 2), round(prob_dist.prob("neg"), 2)
```




    ('pos', 0.63, 0.37)



### Classifying TextBlobs
 Another way to classify text is to pass a classifier into the constructor of TextBlob and call its classify() method.


```python
blob = TextBlob("I lost the battle. But I won the war. Happy ending? Maybe!", classifier=cl)
blob.classify()
```




    'neg'



The advantage of this approach is that you can classify sentences within a TextBlob.


```python
for s in blob.sentences:
    print(s)
    print(s.classify())
```

    I lost the battle.
    neg
    But I won the war.
    neg
    Happy ending?
    pos
    Maybe!
    pos


### Evaluating Classifiers

To compute the accuracy on our test set, use the accuracy(test_data) method.


```python
cl.accuracy(test)
```




    0.8333333333333334



Which are the most informative word features of our classifier?


```python
cl.show_informative_features(10) # Recall that these are the top word features from our original training set
```

    Most Informative Features
                contains(my) = True              neg : pos    =      1.7 : 1.0
                contains(an) = False             neg : pos    =      1.6 : 1.0
                 contains(I) = True              neg : pos    =      1.4 : 1.0
                 contains(I) = False             pos : neg    =      1.4 : 1.0
                contains(my) = False             pos : neg    =      1.3 : 1.0
                contains(ca) = False             pos : neg    =      1.2 : 1.0
          contains(horrible) = False             pos : neg    =      1.2 : 1.0
        contains(restaurant) = False             pos : neg    =      1.2 : 1.0
             contains(these) = False             neg : pos    =      1.2 : 1.0
               contains(not) = False             pos : neg    =      1.2 : 1.0


### VADER

VADER or Valence Aware Dictionary and Sentiment Reasoner is an open-source sentiment analyzer with a pre-built positive/negative/neutral dictionary.

The VADER algorithm outputs sentiment scores to 4 classes of sentiments: positive, negative, neutral, and compound.

* **positive sentiment**: compound score >= 0.05
* **neutral sentiment**: (compound score > -0.05) and (compound score < 0.05)
* **negative sentiment**: compound score <= -0.05

The pos, neu, and neg scores are ratios for proportions of text that fall in each category (so these should add up to 1). These are the most useful metrics if you want multidimensional measures of sentiment for a given sentence.

The compound score is computed by summing the valence scores of each word in the lexicon and then normalizing them to fall **between -1 (most extreme negative) and +1 (most extreme positive)**. This is the most useful metric if you want a single unidimensional measure of sentiment for a given sentence. Calling it a 'normalized, weighted composite score' is accurate.


```python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

vs = sid.polarity_scores("Vader is a cool sentiment analyzer that was built for social media."
                         "Exclamations connote positive sentiment!"
                         "More exclamations mean more positivity!!!!!!" #try adding more '!' to this line and rerunning this cell
                         "Is this a problematic assumption?")
print(vs)
```

    {'neg': 0.129, 'neu': 0.626, 'pos': 0.246, 'compound': 0.5209}


    /opt/conda/lib/python3.6/site-packages/nltk/twitter/__init__.py:20: UserWarning: The twython library has not been installed. Some functionality from the twitter package will not be available.
      warnings.warn("The twython library has not been installed. "



```python
def get_vader_score(text):
    # Polarity score returns dictionary
    sentences = nltk.tokenize.sent_tokenize(text)
    for sent in sentences:
        ss = sid.polarity_scores(sent)
        for k in sorted(ss):
            print('{0}: {1}, '.format(k, ss[k]), end='')
            print()
        
get_vader_score(text)
```

    compound: -0.8591, 
    neg: 0.213, 
    neu: 0.787, 
    pos: 0.0, 
    compound: -0.3182, 
    neg: 0.14, 
    neu: 0.787, 
    pos: 0.073, 
    compound: -0.4588, 
    neg: 0.148, 
    neu: 0.741, 
    pos: 0.11, 


How does VADER's built-in sentiment analyzer compare with TextBlob's built-in sentiment analyzer?


```python
get_vader_score(text), get_textBlob_score(text)
```

    compound: -0.8591, 
    neg: 0.213, 
    neu: 0.787, 
    pos: 0.0, 
    compound: -0.3182, 
    neg: 0.14, 
    neu: 0.787, 
    pos: 0.073, 
    compound: -0.4588, 
    neg: 0.148, 
    neu: 0.741, 
    pos: 0.11, 





    (None, [-1.0, -0.075, -0.05])


