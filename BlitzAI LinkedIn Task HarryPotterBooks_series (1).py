#!/usr/bin/env python
# coding: utf-8

# # IMPORT THE LIBRARIES

# In[1]:


import glob
import nltk
import re
import multiprocessing
from gensim.models.phrases import Phraser, Phrases
from gensim.models import Word2Vec, FastText


# # LOAD THE TEXTS

# In[2]:


book_names = sorted(glob.glob("C:/Users/DELL/Downloads/BiltzAI-Linkedin-nlp engineer job-task/HarryPotterData/*.txt"))
book_names


# # CREATE ONE STRING TOGETHER

# In[3]:


import codecs


# In[4]:


raw_corpus = u""  # will use utf-8
for filename in book_names:
    print("Reading {}...".format(filename.split("/")[1]))
    with codecs.open(filename,"r","utf-8") as book:
        raw_corpus += book.read() 
    print("Corpus now is {} characters and {} words long".format(len(raw_corpus), len(raw_corpus.split())))
    print("~"*30)


# In[5]:


raw_corpus[:5000]


# # TOKENISATION(CORPUS INTO SENTENCES)

# In[6]:


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = tokenizer.tokenize(raw_corpus)


# In[7]:


def sentence_to_words(sentence):
    # remove non leter characters
    cleaned_sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    
    # convert sentence to words
    words = cleaned_sentence.lower().split()
    
    return words
words_corpus = []


# In[8]:


# for now let's leave only words and don't remove stop words
for sentence in sentences:
    if len(sentence) > 0:
        words_corpus.append(sentence_to_words(sentence))


# In[9]:


tokens_number = sum([len(sub_corpus) for sub_corpus in words_corpus])
print('The words corpus contains {0:,} tokens'.format(tokens_number))


# In[10]:


print(words_corpus[:5])


# # FEATURE EXTRACTION

# ### USING WORD2Vec (Define parameter for building Word2Vec model)

# In[11]:


from gensim.models import Word2Vec, FastText
import gensim.models.word2vec as w2v


# In[12]:


# train word2vec model
w2v = Word2Vec(words_corpus, min_count=3, vector_size = 300)
print(w2v)
#Word2Vec(vocab=19, size=5, alpha=0.025)


# In[13]:


len(w2v.wv.index_to_key)


# In[14]:


#list the vocabulary words
words = list(w2v.wv.index_to_key)
print(words)


# # BUILD A VOCAB
w2v.build_vocab(words)
print("Word2Vec vocabulary length:", len(w2v.wv))
# # (Reqirements)

# +You are given all the parts of the Harry Potter Novel Series in form of text files within SemanticSearchData.zip. Your task is to build a module which provides following functionalities
# 
# + An ability to search for any Harry Potter Novel characters, and the search result would include two components
#   + First one is called **RelatedConcepts**: In this section of the search results you'll get all other words or pair of words closer to the search term
#     + Ex: If you search for let's say "Harry Potter", then the **RelatedConcepts** would be a list of top 20 concepts that are close to Harry Potter like [Hermione Granger, Lord Voldermort, Rubeus Hagrid,........]
#   
# + Secondly, you need to provide the book name and list of page numbers of those pages which has the search term.
#     + Ex: Since your search term is Harry Potter, you need to get all the page numbers in all 7 books where this word is mentioned.
# 

# # WORDEMBEDDING(1.SKIP GRAM)

# In[15]:


import time
import os


# In[16]:


if not os.path.exists("trained"):
    os.makedirs("trained")


# In[17]:


start = time.time()
w2v_sg = Word2Vec(sentences=sentences, vector_size=300, window=7, min_count=3,  sg=1, )
print("Training Word2Vec Skip-Gram took {} seconds".format(time.time()-start))


# In[18]:


w2v_sg.save(os.path.join("trained", "w2v_sg.bin"))


# # WORDEMBEDDING(2.CBOW)

# In[19]:


if not os.path.exists("trained"):
    os.makedirs("trained")


# In[20]:


import time
import os


# In[21]:


start = time.time()
w2v_cbow = Word2Vec(sentences=sentences, vector_size=300, window=7, min_count=3, sg=0)
print("Training Word2Vec CBOW took {} seconds".format(time.time()-start))


# In[22]:


w2v_cbow.save(os.path.join("trained", "w2v_cbow.bin"))


# # Exploring the trained models
# 

# # Semantic similarities

# In[23]:


sim_words = w2v.wv.most_similar('harry')
sim_words


# In[24]:


sim_words = w2v.wv.most_similar('harmony')
sim_words


# In[25]:


sim_words = w2v.wv.most_similar('e')
sim_words


# In[26]:


sim_words = w2v.wv.most_similar('tonight')
sim_words


# In[ ]:





# # (reducing dimensions to 2-D for plotting, using T-SNE)

# In[27]:


from sklearn.manifold import TSNE


# In[28]:


tsne = TSNE(n_components=2, random_state=42)


# In[29]:


word_vectors = w2v.wv.vectors


# In[30]:


start = time.time()
word_vectors_2d = tsne.fit_transform(word_vectors)
print("Reducing dimensions to 2-D using T-SNE took {} seconds".format(time.time()-start))


# In[31]:


get_ipython().run_cell_magic('time', '', '\nall_word_vectors_matrix_2d = tsne.fit_transform(word_vectors)')


# # (PLOT THE PICTURE)

# In[43]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
 
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# In[45]:


def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=20).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
     # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))
    


# In[50]:


tsnescatterplot(w2v, 'harry',[t[0] for t in w2v.wv.most_similar(positive=["harry"], topn=20)][10:])


# In[51]:


tsnescatterplot(w2v, 'dursleys',[t[0] for t in w2v.wv.most_similar(positive=["dursleys"], topn=20)][10:])


# # MODEL SAVING

# In[52]:


import pickle

# saving the model 
pickle_out = open("HarryPotterW2V.pkl", mode = "wb") 
pickle.dump(w2v_sg, pickle_out) 
pickle_out.close()


# In[ ]:




