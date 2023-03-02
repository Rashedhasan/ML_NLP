import os
import nltk
import nltk.corpus
#nltk.download()
#tokenization for word seperation
print(os.listdir(nltk.data.find("corpora")))
from nltk.corpus import brown
print(brown.words())
print(nltk.corpus.gutenberg.fileids())
hamlet=nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
print(hamlet)
for word in hamlet[:5000]:
    print(word,sep=' ',end=' ')
AI=""" "AI" redirects here. For other uses, see AI (disambiguation), Artificial intelligence (disambiguation), and Intelligent agent.
Part of a series on
Artificial intelligence
Anatomy-1751201 1280.png
Major goals
Approaches
Philosophy
History
Technology Glossary vte Artificial intelligence (AI) is intelligence—perceiving, synthesizing, and inferring information—demonstrated by machines, as opposed to intelligence displayed by non-human animals and humans. Example tasks in which this is done include speech recognition, computer vision, translation between (natural) languages, as well as other mappings of inputs. AI applications include advanced web search engines (e.g., Google Search), recommendation systems (used by YouTube, Amazon and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo), generative or creative tools (ChatGPT and AI art), automated decision-making and competing at the highest level in strategic game systems (such as chess and Go).[1]
As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.[2] For instance, optical character recognition is frequently excluded from things considered to be AI,[3] having become a routine technology.[4] """
type(AI)
from nltk.tokenize import word_tokenize
AI_tokenize=word_tokenize(AI)
print(AI_tokenize)
len(AI_tokenize)
from nltk.probability import FreqDist
fdist=FreqDist()
for word in AI_tokenize:
    fdist[word.lower()]+=1
print(fdist)
print(fdist['artificial'])
print(len(fdist))
first_top10=fdist.most_common(10)#top 10 frequency words
first_top10
from nltk.tokenize import blankline_tokenize
AI_blank=blankline_tokenize(AI)#paragraph seperation
len(AI_blank)
AI_blank[2]
from nltk.util import bigrams,trigrams,ngrams #groups with 2/3/n words each
string="this is the most beautiful things in the world. it will helps us to monitor everything we want"
quotes_token=nltk.word_tokenize(string)
quotes_token
bigram_token=list(nltk.bigrams(quotes_token))
trigram_token=list(nltk.trigrams(quotes_token))
trigram_token
ngram_token=list(nltk.ngrams(quotes_token,6))
ngram_token
#word stemming affection,affecting,affect,affects to affect mapping
from nltk.stem import PorterStemmer
pst=PorterStemmer()
pst.stem("having")
word_to_stem=["having","giving","given","give"]
for word in word_to_stem:
    print(word+":"+pst.stem(word))#only ing removal
from nltk.stem import LancasterStemmer#more aggressive then porterstemmer SnowballStemmer also has property
pst=LancasterStemmer()
word_to_stem=["having","giving","gave","give"]
for word in word_to_stem:
    print(word+":"+pst.stem(word))
#lemmatization of token that means words to map proper base form go,gone,went to go
from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
word_len=WordNetLemmatizer()
word_len.lemmatize("corpora")
for word in word_to_stem:
    print(word+":"+word_len.lemmatize(word))
from nltk.corpus import stopwords
stopwords.words("english")
len(stopwords.words("english"))
import re
punctuation=re.compile(r'[-.?!,:;()|0-9]')
post_punctuation=[]
for words in AI_tokenize:
    word=punctuation.sub("",words)
    if(len(word)>0):
        post_punctuation.append(word)
#parts of speech processing
sent="Tinothy is a natural when it comes to drawing"
sent_token=word_tokenize(sent)
for token in sent_token:
    print(nltk.pos_tag([token]))#identify parts of speech(pos) tag
#Named entity rocognition(find location,person name, monetary info etc)
from nltk import ne_chunk
NE_sent="The US president stays in the white house"
NE_tokenze=word_tokenize(NE_sent)
NE_tag=nltk.pos_tag(NE_tokenze)
NE_NER=ne_chunk(NE_tag)
print(NE_NER)
#syntax tree
new="the big cat ate the little mouse who was after fresh cheese"
new_token=nltk.pos_tag(word_tokenize(new))
new_token
#chunking group words into meaningful sentence
grammer_np=r"NP: {<DT>?<JJ>*<NN>}"
chunk_parser=nltk.RegexpParser(grammer_np)
chunk_result=chunk_parser.parse(new_token)
chunk_result#error but tree generated