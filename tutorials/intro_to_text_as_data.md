---
layout: page
title: Text as Data
permalink: /tutorials/intro_to_text_as_data/
parent: Tutorials
nav_order: 1
---


# Text as data

How much information is contained in the written word? 

Can we learn about complex political dynamics by parsing texts?

We could attempt to answer these questions using qualitative methods. For example, we could do a deep reading of documents to analyze discourse, identify themes, and draw conclusions. 

But this type of deep reading is hard to scale. Imagine doing a deep read of a single document or a handful of documents; what if we had to read thousands, millions, or billions of texts? Would a single human reader be able to process all of this information?

Rather than attempt to read a billion documents at human speed, we can leverage the statistical properties of text and computational tools to process texts at a much faster rate. 

**This is the "text as data" approach.**

Computational text analysis can reveal a suprising amount of information about the political world, such as [patterns of censorship in an autocratic regime](https://gking.harvard.edu/publications/how-censorship-china-allows-government-criticism-silences-collective-expression), [sources of partisan conflict in Congress](http://languagelog.ldc.upenn.edu/myl/Monroe.pdf), and why [repression sometimes fails to silence dissent](https://www.cambridge.org/core/journals/american-political-science-review/article/abs/how-saudi-crackdowns-fail-to-silence-online-dissent/1BA13DF8FD5D04EC181BCD4D1055254B).

To deploy these tools, however, we need to first understand some basic properties about text objects in Python, starting with strings.


----

## Strings

In Python, text is a **string** object. Let's look at some properties of strings.


```python
sentence = 'a string is a sequence of characters.' 

sentence
```




    'a string is a sequence of characters.'




```python
type(sentence)
```




    str



String objects are sliceable.


```python
sentence[:11]
```




    'A string is'



We can search for characters within strings.


```python
sentence.find('sequence') #where is 'sequence' located in our string object?
```




    14



We can add to a string.


```python
sentence + ' Can I add another sentence?'
```




    'A string is a sequence of characters. Can I add another sentence?'



### Python reads strings differently than humans do

When Python "reads" a string, it reads it literally. That means capitalization, typos, nouns/pronouns matter because they will affect how Python will read characters. 

For example, Python will not see these as equivalent:


```python
'Hark upon the Gale!' == "hark upon the gale!"
```




    False




```python
'even imperceptible differences matter' == 'even imperciptible differences matter'
```




    False



A **huge part** of text analysis involves cleaning (i.e. "pre-processing") strings such that Python will be able to read strings just like a human would.

Let's look at some of these cleaning functions.


```python
sentence.upper()
```




    'A STRING IS A SEQUENCE OF CHARACTERS.'




```python
sentence.lower()
```




    'a string is a sequence of characters.'




```python
sentence.capitalize()
```




    'A string is a sequence of characters.'




```python
sentence.swapcase()
```




    'A STRING IS A SEQUENCE OF CHARACTERS.'



In addition to converting the characters in a string, we can also split our string object into a list. 


```python
sentence.split()
```




    ['A', 'string', 'is', 'a', 'sequence', 'of', 'characters.']



We didn't specify how we wanted to split the string, so it defaulted to splitting on blank spaces. **This creates a list of words.** 

But we could instead specify that we wanted to split on some other text feature, like **punctuation.** Notice that splitting on punctuation changes the **list of words** to a **list of sentences.**


```python
a_few_thoughts = 'Computational text analysis is a pretty powerful tool. The start-up costs are very high, but do not get discouraged'

a_few_thoughts.split('.')
```




    ['Computational text analysis is a pretty powerful tool',
     ' The start-up costs are very high, but do not get discouraged']



Why would we want to split strings like this? 

Strings can range in length from a single character to infinite characters (however many characters are in your corpus). When doing any text analysis, a key pre-processing step is deciding **how you want to split the string** into a **standardized format.** 

The idea is to **create a unit of analysis** -- whether it be a word, a sentence, a paragaph, an article, etc. Whichever unit you choose will vary depending on your research question, your corpus, and your computational end goal.

---

## Pre-processing a document

Let's apply some of these string manipulation tools to pre-process a document: an excerpt from Machiavelli's *The Prince.*

Let's store the excerpt as a string. Note: Triple quotes are used for strings that break across multiple lines


```python
doc = """ Upon this a question arises: whether it be better to be loved than feared or feared than loved? 
It may be answered that one should wish to be both, but, because it is difficult to unite them in one person, 
it is much safer to be feared than loved, when, of the two, either must be dispensed with. Because this is to 
be asserted in general of men, that they are ungrateful, fickle, false, cowardly, covetous, and as long as you
succeed they are yours entirely; they will offer you their blood, property, life, and children, as is said above,
when the need is far distant; but when it approaches they turn against you. And that prince who, relying entirely
on their promises, has neglected other precautions, is ruined; because friendships that are obtained by payments,
and not by greatness or nobility of mind, may indeed be earned, but they are not secured, and in time of need 
cannot be relied upon; and men have less scruple in offending one who is beloved than one who is feared, for love
is preserved by the link of obligation which, owing to the baseness of men, is broken at every opportunity for 
their advantage; but fear preserves you by a dread of punishment which never fails.

Nevertheless a prince ought to inspire fear in such a way that, if he does not win love, he avoids hatred; because 
he can endure very well being feared whilst he is not hated, which will always be as long as he abstains from the
property of his citizens and subjects and from their women. But when it is necessary for him to proceed against the
life of someone, he must do it on proper justification and for manifest cause, but above all things he must keep his
hands off the property of others, because men more quickly forget the death of their father than the loss of their 
patrimony. Besides, pretexts for taking away the property are never wanting; for he who has once begun to live by 
robbery will always find pretexts for seizing what belongs to others; but reasons for taking life, on the contrary, 
are more difficult to find and sooner lapse. But when a prince is with his army, and has under control a multitude 
of soldiers, then it is quite necessary for him to disregard the reputation of cruelty, for without it he would never
hold his army united or disposed to its duties. """


print(doc)
```

     Upon this a question arises: whether it be better to be loved than feared or feared than loved? 
    It may be answered that one should wish to be both, but, because it is difficult to unite them in one person, 
    it is much safer to be feared than loved, when, of the two, either must be dispensed with. Because this is to 
    be asserted in general of men, that they are ungrateful, fickle, false, cowardly, covetous, and as long as you
    succeed they are yours entirely; they will offer you their blood, property, life, and children, as is said above,
    when the need is far distant; but when it approaches they turn against you. And that prince who, relying entirely
    on their promises, has neglected other precautions, is ruined; because friendships that are obtained by payments,
    and not by greatness or nobility of mind, may indeed be earned, but they are not secured, and in time of need 
    cannot be relied upon; and men have less scruple in offending one who is beloved than one who is feared, for love
    is preserved by the link of obligation which, owing to the baseness of men, is broken at every opportunity for 
    their advantage; but fear preserves you by a dread of punishment which never fails.
    
    Nevertheless a prince ought to inspire fear in such a way that, if he does not win love, he avoids hatred; because 
    he can endure very well being feared whilst he is not hated, which will always be as long as he abstains from the
    property of his citizens and subjects and from their women. But when it is necessary for him to proceed against the
    life of someone, he must do it on proper justification and for manifest cause, but above all things he must keep his
    hands off the property of others, because men more quickly forget the death of their father than the loss of their 
    patrimony. Besides, pretexts for taking away the property are never wanting; for he who has once begun to live by 
    robbery will always find pretexts for seizing what belongs to others; but reasons for taking life, on the contrary, 
    are more difficult to find and sooner lapse. But when a prince is with his army, and has under control a multitude 
    of soldiers, then it is quite necessary for him to disregard the reputation of cruelty, for without it he would never
    hold his army united or disposed to its duties. 



```python
type(doc)
```




    str




```python
doc = doc.lower()
doc # notice that the new lines in the block quote have been converted to \n
```




    ' upon this a question arises: whether it be better to be loved than feared or feared than loved? \nit may be answered that one should wish to be both, but, because it is difficult to unite them in one person, \nit is much safer to be feared than loved, when, of the two, either must be dispensed with. because this is to \nbe asserted in general of men, that they are ungrateful, fickle, false, cowardly, covetous, and as long as you\nsucceed they are yours entirely; they will offer you their blood, property, life, and children, as is said above,\nwhen the need is far distant; but when it approaches they turn against you. and that prince who, relying entirely\non their promises, has neglected other precautions, is ruined; because friendships that are obtained by payments,\nand not by greatness or nobility of mind, may indeed be earned, but they are not secured, and in time of need \ncannot be relied upon; and men have less scruple in offending one who is beloved than one who is feared, for love\nis preserved by the link of obligation which, owing to the baseness of men, is broken at every opportunity for \ntheir advantage; but fear preserves you by a dread of punishment which never fails.\n\nnevertheless a prince ought to inspire fear in such a way that, if he does not win love, he avoids hatred; because \nhe can endure very well being feared whilst he is not hated, which will always be as long as he abstains from the\nproperty of his citizens and subjects and from their women. but when it is necessary for him to proceed against the\nlife of someone, he must do it on proper justification and for manifest cause, but above all things he must keep his\nhands off the property of others, because men more quickly forget the death of their father than the loss of their \npatrimony. besides, pretexts for taking away the property are never wanting; for he who has once begun to live by \nrobbery will always find pretexts for seizing what belongs to others; but reasons for taking life, on the contrary, \nare more difficult to find and sooner lapse. but when a prince is with his army, and has under control a multitude \nof soldiers, then it is quite necessary for him to disregard the reputation of cruelty, for without it he would never\nhold his army united or disposed to its duties. '



We can edit the string. For example, let's replace every instance of 'he' with 'prince'. 

Why might we want to do this if we were doing a computational analysis of _The Prince_?


```python
doc.replace('he','prince') # what happened here?
```




    ' upon this a question arises: wprincetprincer it be better to be loved than feared or feared than loved? \nit may be answered that one should wish to be both, but, because it is difficult to unite tprincem in one person, \nit is much safer to be feared than loved, wprincen, of tprince two, eitprincer must be dispensed with. because this is to \nbe asserted in general of men, that tprincey are ungrateful, fickle, false, cowardly, covetous, and as long as you\nsucceed tprincey are yours entirely; tprincey will offer you tprinceir blood, property, life, and children, as is said above,\nwprincen tprince need is far distant; but wprincen it approacprinces tprincey turn against you. and that prince who, relying entirely\non tprinceir promises, has neglected otprincer precautions, is ruined; because friendships that are obtained by payments,\nand not by greatness or nobility of mind, may indeed be earned, but tprincey are not secured, and in time of need \ncannot be relied upon; and men have less scruple in offending one who is beloved than one who is feared, for love\nis preserved by tprince link of obligation which, owing to tprince baseness of men, is broken at every opportunity for \ntprinceir advantage; but fear preserves you by a dread of punishment which never fails.\n\nnevertprinceless a prince ought to inspire fear in such a way that, if prince does not win love, prince avoids hatred; because \nprince can endure very well being feared whilst prince is not hated, which will always be as long as prince abstains from tprince\nproperty of his citizens and subjects and from tprinceir women. but wprincen it is necessary for him to proceed against tprince\nlife of someone, prince must do it on proper justification and for manifest cause, but above all things prince must keep his\nhands off tprince property of otprincers, because men more quickly forget tprince death of tprinceir fatprincer than tprince loss of tprinceir \npatrimony. besides, pretexts for taking away tprince property are never wanting; for prince who has once begun to live by \nrobbery will always find pretexts for seizing what belongs to otprincers; but reasons for taking life, on tprince contrary, \nare more difficult to find and sooner lapse. but wprincen a prince is with his army, and has under control a multitude \nof soldiers, tprincen it is quite necessary for him to disregard tprince reputation of cruelty, for without it prince would never\nhold his army united or disposed to its duties. '



Strings are case sensitive and **space sensitive**. In this case, the string replace function looked within `doc` for every instance of 'he' and replaced it with 'prince', including when 'he' was within another word.

Hence __"whether"__  h<sub>&rightarrow;</sub>(x) __"wprincetprincer"__

One way to prevent this is to add spaces in our replace function.


```python
doc.replace(' he ',' prince ')
```




    ' upon this a question arises: whether it be better to be loved than feared or feared than loved? \nit may be answered that one should wish to be both, but, because it is difficult to unite them in one person, \nit is much safer to be feared than loved, when, of the two, either must be dispensed with. because this is to \nbe asserted in general of men, that they are ungrateful, fickle, false, cowardly, covetous, and as long as you\nsucceed they are yours entirely; they will offer you their blood, property, life, and children, as is said above,\nwhen the need is far distant; but when it approaches they turn against you. and that prince who, relying entirely\non their promises, has neglected other precautions, is ruined; because friendships that are obtained by payments,\nand not by greatness or nobility of mind, may indeed be earned, but they are not secured, and in time of need \ncannot be relied upon; and men have less scruple in offending one who is beloved than one who is feared, for love\nis preserved by the link of obligation which, owing to the baseness of men, is broken at every opportunity for \ntheir advantage; but fear preserves you by a dread of punishment which never fails.\n\nnevertheless a prince ought to inspire fear in such a way that, if prince does not win love, prince avoids hatred; because \nhe can endure very well being feared whilst prince is not hated, which will always be as long as prince abstains from the\nproperty of his citizens and subjects and from their women. but when it is necessary for him to proceed against the\nlife of someone, prince must do it on proper justification and for manifest cause, but above all things prince must keep his\nhands off the property of others, because men more quickly forget the death of their father than the loss of their \npatrimony. besides, pretexts for taking away the property are never wanting; for prince who has once begun to live by \nrobbery will always find pretexts for seizing what belongs to others; but reasons for taking life, on the contrary, \nare more difficult to find and sooner lapse. but when a prince is with his army, and has under control a multitude \nof soldiers, then it is quite necessary for him to disregard the reputation of cruelty, for without it prince would never\nhold his army united or disposed to its duties. '



## Removing stop words and punctuation

Let's do so simple pre-processing to remove stop words, punctuation, and other characters that might not be relevant to our analysis.


```python
stop_words = ['the', 'it', 'is', 'a', 'was', 'and', 
             'why', 'what', 'how', 'has', 'have', 'this', 'that']

punctuation = [".", "," , "?", "!", "#", "$", '\n']
```


```python
for p in punctuation:
    doc = doc.replace(p,'') # erase punctuation

words = doc.split() # store the processed doc in an object called 'words'

# remove stopwords and store remaining words in a new list
results = []
for word in words:
    if word not in stop_words:
        results.append(word)

print(results)
```

    ['upon', 'question', 'arises:', 'whether', 'be', 'better', 'to', 'be', 'loved', 'than', 'feared', 'or', 'feared', 'than', 'loved', 'may', 'be', 'answered', 'one', 'should', 'wish', 'to', 'be', 'both', 'but', 'because', 'difficult', 'to', 'unite', 'them', 'in', 'one', 'person', 'much', 'safer', 'to', 'be', 'feared', 'than', 'loved', 'when', 'of', 'two', 'either', 'must', 'be', 'dispensed', 'with', 'because', 'to', 'be', 'asserted', 'in', 'general', 'of', 'men', 'they', 'are', 'ungrateful', 'fickle', 'false', 'cowardly', 'covetous', 'as', 'long', 'as', 'yousucceed', 'they', 'are', 'yours', 'entirely;', 'they', 'will', 'offer', 'you', 'their', 'blood', 'property', 'life', 'children', 'as', 'said', 'abovewhen', 'need', 'far', 'distant;', 'but', 'when', 'approaches', 'they', 'turn', 'against', 'you', 'prince', 'who', 'relying', 'entirelyon', 'their', 'promises', 'neglected', 'other', 'precautions', 'ruined;', 'because', 'friendships', 'are', 'obtained', 'by', 'paymentsand', 'not', 'by', 'greatness', 'or', 'nobility', 'of', 'mind', 'may', 'indeed', 'be', 'earned', 'but', 'they', 'are', 'not', 'secured', 'in', 'time', 'of', 'need', 'cannot', 'be', 'relied', 'upon;', 'men', 'less', 'scruple', 'in', 'offending', 'one', 'who', 'beloved', 'than', 'one', 'who', 'feared', 'for', 'loveis', 'preserved', 'by', 'link', 'of', 'obligation', 'which', 'owing', 'to', 'baseness', 'of', 'men', 'broken', 'at', 'every', 'opportunity', 'for', 'their', 'advantage;', 'but', 'fear', 'preserves', 'you', 'by', 'dread', 'of', 'punishment', 'which', 'never', 'failsnevertheless', 'prince', 'ought', 'to', 'inspire', 'fear', 'in', 'such', 'way', 'if', 'he', 'does', 'not', 'win', 'love', 'he', 'avoids', 'hatred;', 'because', 'he', 'can', 'endure', 'very', 'well', 'being', 'feared', 'whilst', 'he', 'not', 'hated', 'which', 'will', 'always', 'be', 'as', 'long', 'as', 'he', 'abstains', 'from', 'theproperty', 'of', 'his', 'citizens', 'subjects', 'from', 'their', 'women', 'but', 'when', 'necessary', 'for', 'him', 'to', 'proceed', 'against', 'thelife', 'of', 'someone', 'he', 'must', 'do', 'on', 'proper', 'justification', 'for', 'manifest', 'cause', 'but', 'above', 'all', 'things', 'he', 'must', 'keep', 'hishands', 'off', 'property', 'of', 'others', 'because', 'men', 'more', 'quickly', 'forget', 'death', 'of', 'their', 'father', 'than', 'loss', 'of', 'their', 'patrimony', 'besides', 'pretexts', 'for', 'taking', 'away', 'property', 'are', 'never', 'wanting;', 'for', 'he', 'who', 'once', 'begun', 'to', 'live', 'by', 'robbery', 'will', 'always', 'find', 'pretexts', 'for', 'seizing', 'belongs', 'to', 'others;', 'but', 'reasons', 'for', 'taking', 'life', 'on', 'contrary', 'are', 'more', 'difficult', 'to', 'find', 'sooner', 'lapse', 'but', 'when', 'prince', 'with', 'his', 'army', 'under', 'control', 'multitude', 'of', 'soldiers', 'then', 'quite', 'necessary', 'for', 'him', 'to', 'disregard', 'reputation', 'of', 'cruelty', 'for', 'without', 'he', 'would', 'neverhold', 'his', 'army', 'united', 'or', 'disposed', 'to', 'its', 'duties']



```python
# list comprehension achieves same result as above in a single line

list_comp = [vocab for vocab in words if vocab not in punctuation and vocab not in stop_words]

list_comp == results
```




    True



## Labeling texts

After making these simple modifications to the text, we can start doing some analysis on it. Like labeling sentiment.


```python
positive_words = ['loved','greatness', 'advantage', 'opportunity', 'friendship']
negative_words = ['feared', 'ungrateful', 'fickle', 'false', 'cowardly', 'covetous'] 
        
len([w for w in words if w in positive_words])
```




    5




```python
len([w for w in words if w in negative_words])
```




    10



10 negative words versus 5 positive words...

Based on our very breezy analysis, Machiaveli seems pretty negative.

---

## A Better Way: NLTK

I've just shown you how to do basic pre-processing steps from scratch. When you're learning how to do computational text analysis, it's a good idea to start by writing your own functions so you understand exactly what steps you're taking and how you're editing your text. 

But once you get the hang of things, there are many existing packages to help you efficiently clean and prepare your text. Let's look at one of the most widely used ones: the Natural Language Tool Kit (NLTK). 

NLTK is a great resource for getting started with basic text analysis because it has [excellent documentation.](http://www.nltk.org/book/)


```python
# NLTK: The Natural Language Tool Kit

import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg

gutenberg.fileids() # these are the corpora included with nltk -- very good for practice!
```

    [nltk_data] Downloading package gutenberg to
    [nltk_data]     /Users/fionashenbayh/nltk_data...
    [nltk_data]   Package gutenberg is already up-to-date!





    ['austen-emma.txt',
     'austen-persuasion.txt',
     'austen-sense.txt',
     'bible-kjv.txt',
     'blake-poems.txt',
     'bryant-stories.txt',
     'burgess-busterbrown.txt',
     'carroll-alice.txt',
     'chesterton-ball.txt',
     'chesterton-brown.txt',
     'chesterton-thursday.txt',
     'edgeworth-parents.txt',
     'melville-moby_dick.txt',
     'milton-paradise.txt',
     'shakespeare-caesar.txt',
     'shakespeare-hamlet.txt',
     'shakespeare-macbeth.txt',
     'whitman-leaves.txt']



In the Machiavellian vein, let's focus on Shakespeare's famous prince: *Hamlet.*


```python
hamlet = gutenberg.raw('shakespeare-hamlet.txt') #yes this is the full text of Hamlet

hamlet[:1000] #let's just print the first 1000 characters
```




    "[The Tragedie of Hamlet by William Shakespeare 1599]\n\n\nActus Primus. Scoena Prima.\n\nEnter Barnardo and Francisco two Centinels.\n\n  Barnardo. Who's there?\n  Fran. Nay answer me: Stand & vnfold\nyour selfe\n\n   Bar. Long liue the King\n\n   Fran. Barnardo?\n  Bar. He\n\n   Fran. You come most carefully vpon your houre\n\n   Bar. 'Tis now strook twelue, get thee to bed Francisco\n\n   Fran. For this releefe much thankes: 'Tis bitter cold,\nAnd I am sicke at heart\n\n   Barn. Haue you had quiet Guard?\n  Fran. Not a Mouse stirring\n\n   Barn. Well, goodnight. If you do meet Horatio and\nMarcellus, the Riuals of my Watch, bid them make hast.\nEnter Horatio and Marcellus.\n\n  Fran. I thinke I heare them. Stand: who's there?\n  Hor. Friends to this ground\n\n   Mar. And Leige-men to the Dane\n\n   Fran. Giue you good night\n\n   Mar. O farwel honest Soldier, who hath relieu'd you?\n  Fra. Barnardo ha's my place: giue you goodnight.\n\nExit Fran.\n\n  Mar. Holla Barnardo\n\n   Bar. Say, what is Horatio there?\n  Hor. A peece of"



One of the very nice things about nltk is that it has ready-built functions to standardize our texts.


```python
hamlet_words = gutenberg.words('shakespeare-hamlet.txt') #list of words
hamlet_sents = gutenberg.sents('shakespeare-hamlet.txt') # list of sentences, each sentence is list
hamlet_paras = gutenberg.paras('shakespeare-hamlet.txt') # list of pararaphs, each paragraph is a list
```


```python
hamlet_words[:5]
```

    ['[', 'The', 'Tragedie', 'of', 'Hamlet']



```python
hamlet_sents[:5]
```

    [['[', 'The', 'Tragedie', 'of', 'Hamlet', 'by', 'William', 'Shakespeare', '1599', ']'], ['Actus', 'Primus', '.'], ['Scoena', 'Prima', '.'], ['Enter', 'Barnardo', 'and', 'Francisco', 'two', 'Centinels', '.'], ['Barnardo', '.']]



```python
hamlet_paras[:5]
```




    [[['[',
       'The',
       'Tragedie',
       'of',
       'Hamlet',
       'by',
       'William',
       'Shakespeare',
       '1599',
       ']']],
     [['Actus', 'Primus', '.'], ['Scoena', 'Prima', '.']],
     [['Enter', 'Barnardo', 'and', 'Francisco', 'two', 'Centinels', '.']],
     [['Barnardo', '.'],
      ['Who', "'", 's', 'there', '?'],
      ['Fran', '.'],
      ['Nay', 'answer', 'me', ':', 'Stand', '&', 'vnfold', 'your', 'selfe']],
     [['Bar', '.'], ['Long', 'liue', 'the', 'King']]]




```python
from nltk.corpus import stopwords
stopwords.words('english')[:20] #the brackets are to index the first 20 items in the list
```




    ['i',
     'me',
     'my',
     'myself',
     'we',
     'our',
     'ours',
     'ourselves',
     'you',
     "you're",
     "you've",
     "you'll",
     "you'd",
     'your',
     'yours',
     'yourself',
     'yourselves',
     'he',
     'him',
     'his']




```python
[word for word in hamlet_words if word.lower() not in stopwords.words('english')][:10] # first 10 items in the list
```




    ['[',
     'Tragedie',
     'Hamlet',
     'William',
     'Shakespeare',
     '1599',
     ']',
     'Actus',
     'Primus',
     '.']




```python
[word.lower() for word in hamlet_words if word.isalpha()][:10] 
```




    ['the',
     'tragedie',
     'of',
     'hamlet',
     'by',
     'william',
     'shakespeare',
     'actus',
     'primus',
     'scoena']




```python
from nltk.stem.porter import PorterStemmer
porter = nltk.PorterStemmer()

print([word for word in hamlet_words[80:110]])
print([porter.stem(word) for word in hamlet_words[80:110]])
```

    ['releefe', 'much', 'thankes', ':', "'", 'Tis', 'bitter', 'cold', ',', 'And', 'I', 'am', 'sicke', 'at', 'heart', 'Barn', '.', 'Haue', 'you', 'had', 'quiet', 'Guard', '?', 'Fran', '.', 'Not', 'a', 'Mouse', 'stirring', 'Barn']
    ['releef', 'much', 'thank', ':', "'", 'ti', 'bitter', 'cold', ',', 'and', 'I', 'am', 'sick', 'at', 'heart', 'barn', '.', 'haue', 'you', 'had', 'quiet', 'guard', '?', 'fran', '.', 'not', 'a', 'mous', 'stir', 'barn']


One of the best things about NLTK is that its tools can be applied to any document or corpus -- not just the ones in the Gutenberg corpus.


```python
nltk.word_tokenize("I am gonna tokenize this sentence into a list of words. For fun.")
```




    ['I',
     'am',
     'gon',
     'na',
     'tokenize',
     'this',
     'sentence',
     'into',
     'a',
     'list',
     'of',
     'words',
     '.',
     'For',
     'fun',
     '.']


