#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[2]:


from nltk.corpus import stopwords


# In[3]:


nltk.download('stopwords')


# In[4]:


stopword_nltk = stopwords.words('english')


# In[5]:


print(stopword_nltk)


# In[6]:


print(len(stopword_nltk))


# In[7]:


text="Florance Nightingale was a nurse who lives in the 19th century.She was named after the city of Florance in Italy,where her parents went after they got married in 1818."


# In[8]:


word=[word for word in text.split() if word.lower() not in stopword_nltk]


# In[9]:


print(word)


# In[10]:


new_text="".join(word)


# In[11]:


print(new_text)


# In[12]:


print(len(text))


# In[13]:


pip install spacy


# In[14]:


pip intsall en_core_web_sm


# In[15]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[23]:


import spacy


# In[24]:


en=spacy.load('en_core_web_sm')


# In[18]:


stopword_spacy=en.Defaults.stop_words


# In[19]:


print(en.tokenizer('Florance Nightingale was a nurse who lives in the  century'))


# In[21]:


words=[word for word in text.split() if word.lower() not in stopword_nltk]


# In[22]:


new_text="".join(words)


# In[ ]:


print(new_)

