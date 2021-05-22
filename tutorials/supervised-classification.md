# Supervised Classification

Classification is the task of choosing the 'correct' label for an input. Supervised classification refers to a labeling task where the **labels** are **defined in advance** (this is in contrast to unsupervised classification, e.g. topic modeling, where the labels are not predefined). Some examples of supervised classification include:

* Categorizing an email as spam or not spam
* Categorizing the topic of a news article is from a fixed list of topics
* Categorizing the sentiment of a document as positive, negative, or neutral

**Supervised classification** tasks uses data that has *already* been classified in order to train machine learning algorithms to assign labels to unclassified data. The central idea here is that we are training our computer to look for certain word features of text to develop a model of language labels. This model can then be used to classify new bodies of text.

Below, we will explore different supervised classification methods. 

### Gender classification

Names ending in a, e and i are likely to be female, while names ending in k, o, r, s and t are likely to be male. Let's build a classifier to model these differences more precisely.

The first step in creating a classifier is deciding what features of the input are relevant, and how to encode those features. For this example, we'll start by just looking at the final letter of a given name. The following feature extractor function builds a dictionary containing relevant information about a given name.


```python
def gender_features(word):
    return {'last_letter': word[-1]}

gender_features('Louis')
```

The output of our gender_features function is a dictionary of feature sets, which maps feature names (last_letter) to their values (word[-1]). Feature names typically provide a human-readable description of the feature, as in the example 'last_letter'. Feature values are typically simple values, such as booleans, numbers, or strings. In this case, it is a simple string.

Now that we've defined a gender feature extractor, we need to prepare a list of examples and corresponding class labels.


```python
import nltk
from nltk.corpus import names
labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])

import random
random.shuffle(labeled_names)

labeled_names[:10] #list first 10 names
```

We have just created a list of word features with gender labels (stored in the object **labeled_names**). 

What does our features set look like?


```python
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
featuresets[:10] 
```

Let's now divide this features list into two sets: a **training set** and a **test set**. The training set will be used to train a computer algorithm classifier. The test set will be used to evaluate how well our classifier performs.


```python
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
```

Let's test how our classifier works on names that didn't appear in the training or test set


```python
classifier.classify(gender_features('Gandalf')), classifier.classify(gender_features('Bilbo'))
```

These character names from *The Hobbit* are correctly classified. But our classifier isn't perfect.


```python
classifier.classify(gender_features('Katniss')), classifier.classify(gender_features('Dumbledore')) 
```

A classifier will never be 100% accurate. But we can systematically evaluate how well a classifier performs by looking at how well it classifies data that has already been labeled.

Below, let's look at how our classifier assigns gender labels to data in our test set. Discrepancies between the gender label assigned by our classifier and the gender labels that were already included in the test set provides a measure of classifier accuracy.


```python
print(nltk.classify.accuracy(classifier, test_set))
```

Our gender classifier is approximately 75% accurate, which is pretty good.

We can further examine the classifier to determine which features it found most effective for distinguishing gender.


```python
classifier.show_most_informative_features(5)
```

This list shows the likelihood ratios between different word features and their labeled categories. For example, names in the training set that end in "a" are about 36 times more likely to be female than male, but names that end in "k" are 32 times more likely to be male than female.

# Choosing The Right Features

Selecting relevant features and deciding how to encode them for a classifier can have an enormous impact on the classifier's ability to extract a good model. Much of the interesting work in building a classifier is deciding what features might be relevant and how best to represent them. Although it's often possible to get decent performance by using a fairly simple and obvious set of features, there are usually significant gains to be had by using carefully constructed features based on an understanding of the task at hand.

Typically, feature extractors are built through a process of trial-and-error. It's common to start with a "kitchen sink" approach and then checking to see which features actually are helpful.


```python
def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features

gender_features2('Tommy') 
```

However, there are limits to the number of features that you should use. Too many features can make the algorithm rely on idiosyncrasies in your training data that don't generalize well to new examples. This problem is known as ***overfitting***, and can be especially problematic when working with small training sets. 

How does gender_features2 compare against our original gender_features classifier?


```python
featuresets = [(gender_features2(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
```

### Error analysis

Once an initial set of features has been chosen, we can refine the feature set using **error analysis**. 

First, we select a development set, containing the corpus data for creating the model. This development set is then subdivided into the training set and the dev-test set.


```python
train_names = labeled_names[1500:]
devtest_names = labeled_names[500:1500]
test_names = labeled_names[:500]
```

The training set is used to train the model and the dev-test set is used to perform error analysis. Note: it is important that we employ a separate dev-test set for error analysis rather than just using the test set. The division of the corpus data into different subsets is shown below.

![image.png](attachment:image.png)

We train a model using the training set [1], and then run it on the dev-test set [2]:


```python
train_set = [(gender_features2(n), gender) for (n, gender) in train_names]
devtest_set = [(gender_features2(n), gender) for (n, gender) in devtest_names]
test_set = [(gender_features2(n), gender) for (n, gender) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set) #[1]
print(nltk.classify.accuracy(classifier, devtest_set)) #[2]
```

Using the dev-test set, we can generate a list of errors that the classifier makes when predicting name genders:


```python
errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features2(name))
    if guess != tag:
        errors.append( (tag, guess, name) )
```

We can then examine individual error cases where the model predicted the wrong label and try to determine what additional pieces of information would allow the classifier to make the right decision (or which existing pieces of information are tricking it into making the wrong decision). The feature set can then be adjusted accordingly.


```python
len(errors) # number of mislabeled names
```


```python
for (tag, guess, name) in sorted(errors):
    print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name))
```

Looking through this list of errors, it appears that certain 2-character suffixes are indicative of gender. For example, names ending in 'yn' appear to be predominantly female, despite the fact that names ending in 'n' tend to be male; and names ending in 'ch' are usually male, even though names that end in 'h' tend to be female. 

Let's adjust our feature extractor to include features for two-letter suffixes:


```python
def gender_features(word):
    return {'suffix1': word[-1:], 'suffix2': word[-2:]}
```

Let's rebuild our classifier with the new feature extractor and see whether its accuracy improves.


```python
train_set = [(gender_features(n), gender) for (n, gender) in train_names]
devtest_set = [(gender_features(n), gender) for (n, gender) in devtest_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, devtest_set))
```

This error analysis procedure can be repeated after checking for patterns in errors made by the improved classifier. *Each time the error analysis procedure is repeated, we should select a different dev-test/training split, to ensure that the classifier does not start to reflect idiosyncrasies in the dev-test set*.

However, once we've used the dev-test set to develop the model, we can no longer trust that it will give us an accurate idea of how well the model would perform on new data. *It is therefore important to keep the test set separate, and unused, until our model development is complete*. At that point, we can use the test set to evaluate how well our model will perform on new input values.

### Exercise

In addition to the suffix features, add another word feature to your gender classifier and evaluate the accuracy of this new model.

# Document Classification

Sometimes we want to classify an entire document rather than a particular word. Document classification is especially useful for sentiment analysis because the emotions or opinions contained in a text are not typically represented by a single word. 

Sentiment analysis is a category of supervised classification algorithms where the labels consist of different emotions, often negative, positive, and neutral. The procedure is the same as before: using pre-labeled corpora, we can build classifiers that automatically tag new documents with appropriate category labels.

As an illustrative example, let's work with the Movie Reviews Corpus that categorizes each review as positive or negative.


```python
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
```

Now let's define a feature extractor for our documents such that the classifier will know which aspects of the data it should pay attention to.

For document topic identification, we can define a feature for each word that indicates whether the document contains that word. To limit the number of features that the classifier needs to process, we begin by constructing a list of the 2000 most frequent words in the overall corpus. We can then define a feature extractor that simply checks whether each of these words is present in a given document.


```python
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
```

Now that we've defined our feature extractor we can use it to train a classifier to label new movie reviews. To check how reliable the resulting classifier is, we compute its accuracy on the test set. And once again, we can use show_most_informative_features() to find out which features the classifier found to be most informative.


```python
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
```


```python
print(nltk.classify.accuracy(classifier, test_set))
```


```python
classifier.show_most_informative_features(10)
```

# Example using Naive Bayes Classification

Let's run some classification tasks using the Supreme Court confirmation hearings transcripts, found here: [https://www.rstreet.org/2019/04/04/supreme-court-confirmation-hearing-transcripts-as-data/](http://)

This data contains the text of every Supreme Court confirmation hearing for which Senate Judiciary Committee  transcripts are available (beginning in 1971 with hearings for Lewis Powell and William Rehnquist and concluding with Neil Gorsuch’s 2017 hearing). 


```python
import pandas as pd
import nltk

data = pd.read_csv('../input/Oct-7-2019-Supreme-Court-Confirmation-Hearing-Transcript.csv', encoding='latin1')

data = data.reset_index()

data.head()
```

### Classifying partisanship

Do Republicans and Democrats speak differently at judicial confirmation hearings? That is, can we infer party label based on what a speaker says? 

The dataset already includes the party label of each speaker. We can use this information to create a partisanship classifier.

The first thing we need to do to create our classifier is to create a set of word features associated with a given party label. There are a few pre-processing steps we will need to do in order to extract the labeled text features from our pandas dataframe object so it can be added to our classifier.

### Pre-processing steps

The first thing we want to do is create a list of all words across all documents. Recall that this information is used to calculate the posterior probabilities in naive bayes classification.

Let's begin by tokenizing all of the documents.


```python
words = data['Statements'].apply(nltk.word_tokenize)
```


```python
words # the tokenized words are structured as a dataframe.
```

Turn this dataframe into a list object.


```python
words = list(words) # the tokenized words are now structured as a list of lists
```

Create a master list of **all words ** across **all documents**. This list will be used to construct our features set below.


```python
import itertools
all_words = (list(itertools.chain.from_iterable(words))) # we will use this information below when we build our naive bayes classifier.
```

Now let's pre-process the text in the original dataframe, tokenizing words and converting everything to lower case.


```python
data['Statements'] = data['Statements'].apply(lambda x: x.lower())
data['Statements'] = data['Statements'].apply(nltk.word_tokenize)
```

Create a list object that links texts with party label. We will split this data up into a training and test set for our classifier.


```python
documents = list(zip(data['Statements'], data['Speaker (Party)(or nominated by)']))
```

Now let's use this information to create a features extractor for our documents. The following features extractor indicates whether a given document contains a word from our master lists of words (all_words). To limit the number of features that the classifier needs to process, we begin by constructing a list of the 2000 most frequent words in the overall corpus [1]. We then define a feature extractor [2] that simply checks whether each of these words is present in a given document.


```python
all_words = nltk.FreqDist(w.lower() for w in all_words) #[1]
word_features = list(all_words)[:2000]

def document_features(document): #[2]
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
```

Now that we've defined our feature extractor, we can use it to train a classifier to label new, previously unseen texts. To check how reliable the resulting classifier is, we compute its accuracy on the test set. And once again, we can use the command show_most_informative_features( ) to find out which features the classifier found to be most informative.


```python
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
```


```python
classifier.show_most_informative_features(5)
```


```python
print(nltk.classify.accuracy(classifier, test_set))
```

With approximately 35% accuracy, this is a pretty abysmal classifier. This classifier performs more poorly than if we just flipped a coin to assign party label (i.e. 50% accuracy)!

What could we do to improve our classifier? How about we restrict our sample to a single confirmation hearing, i.e. Neil Gorsuch's hearing in 2017? How well does the classifier perform for this subset?


```python
data = pd.read_csv('../input/Oct-7-2019-Supreme-Court-Confirmation-Hearing-Transcript.csv', encoding='latin1')
data = data.reset_index()


data = data[data['Statements'].notnull() & (data['Year'] == 2017)] # only hearings from 2017
words = data['Statements'].apply(nltk.word_tokenize)
words = list(words)

all_words = (list(itertools.chain.from_iterable(words)))


data['Statements'] = data['Statements'].apply(lambda x: x.lower())
data['Statements'] = data['Statements'].apply(nltk.word_tokenize)


documents = list(zip(data['Statements'], data['Speaker (Party)(or nominated by)']))

all_words = nltk.FreqDist(w.lower() for w in all_words)
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))

```

Restricting our classifier to 2017 improved our accuracy by over 50%! Why do you think the classifier does a better job on 2017 than for the corpus as a whole? 

# Exercises

* Using the sentiment labels you constructed for the Supreme Court Confirmation Hearings transcripts, use a Naive Bayes classifier to predict the sentiment of statements made by and before the Senate Judiciary Committee.
* Evaluate the accuracy of your classifier.
* Run error analysis on your classifier. Which features are contributing to misclassification?
* Try improving the accuracy of your classifier by adding or substracting word features or subsetting the data.

