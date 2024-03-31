#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup


# In[2]:


import requests


# In[3]:


a='https://www.worldometers.info/coronavirus/'


# In[4]:


html=requests.get(a)


# In[5]:


print(html.text)


# In[6]:


soup=BeautifulSoup(html.text,'lxml')


# In[7]:


from bs4 import BeautifulSoup


# In[8]:


print(soup.prettify())


# In[9]:


soup.h1


# In[10]:


header_h1 =soup.find_all(Id="maincounter-wrap")


# In[11]:


for head_h1 in header_h1:
    print(head_h1.h1.contents[0])
    print(head_h1.div.span.contents[0], end="\n"*2)


# In[ ]:





# In[12]:


scrp_table=soup.find('table',id='main_table_countries_today')


# In[13]:


scrp_table


# In[14]:


headers=[]
for i in scrp_table.find_all('th'):
    title=i.text
    headers.append(title)


# In[17]:


print(headers)


# In[16]:


print(title)


# In[19]:


headers[10]


# In[20]:


headers[13]


# In[24]:


import pandas as pd
scrapdata=pd.DataFrame(columns=headers)


# In[26]:


for tr in scrp_table.find_all('th')[1:]:
    row_data=tr.find_all('td')
    row=[td.text for td in row_data]
    length=len(scrapdata)
    scrapdata.loc[length]=row


# In[ ]:


scrapdata.drop(scrapdata.index[0:8], inplace=True)
scrapdata.drop(scrapdata.index[228:236],inplace=True)
scrapdata.reset_index(inplace=True, drop True)

