#!/usr/bin/env python
# coding: utf-8

# import nltk
# nltk.download()

# In[2]:


dir(nltk)


# In[3]:


from nltk.corpus import stopwords
stopwords.words('english')[0:500:25]


# In[4]:


rawData = open("SMSSpamCollection.tsv").read()


# In[5]:


print(rawData[0:500])


# In[6]:


parsedData = rawData.replace('\t','\n').split('\n')


# In[7]:


parsedData[0:5]


# In[8]:


labelList = parsedData[0::2]
textList = parsedData[1::2]


# In[9]:


print(labelList[0:5])
print(textList[0:5])


# In[10]:


print(len(labelList))
print(len(textList))


# In[11]:


print(labelList[-5:])


# In[12]:


import pandas as pd

fullCorpus = pd.DataFrame({
    'label': labelList[:-1],
    'body_list': textList
})


# In[13]:


fullCorpus.head()


# In[14]:


dataSet = pd.read_csv("SMSSpamCollection.tsv", sep="\t", header=None)


# In[15]:


dataSet.head()


# In[16]:


fullCorpus = pd.read_csv("SMSSpamCollection.tsv", sep="\t", header=None)


# In[17]:


fullCorpus.head()


# In[18]:


fullCorpus.columns = ['label','body_list']


# In[19]:


fullCorpus.head()


# In[20]:


print("Input data has {} rows and {} columns".format(len(fullCorpus),len(fullCorpus.columns)))


# In[21]:


print("Out of {} rows, {} are spams ,{} are hams".format(len(fullCorpus),
                                                        len(fullCorpus[fullCorpus['label']=='spam']),
                                                        len(fullCorpus[fullCorpus['label']=='ham'])))


# In[22]:


print("Number of nulls in label : {}".format(fullCorpus['label'].isnull().sum()))
print("Number of nulls in body_list : {}".format(fullCorpus['body_list'].isnull().sum()))


# In[23]:


import re
re_test = "This is a made up string to test 2 different regex methods"
re_test_messy = "This      is a made up      string to test 2     different regex methods"
re_test_messy1 = 'This-is-a-made/up.string*to>>>>test----2""""""different-regex-methods'


# In[24]:


re.split('\s',re_test)


# In[25]:


re.split('\s',re_test_messy)


# In[26]:


re.split('\s+',re_test_messy)


# In[27]:


re.split('\s+',re_test_messy1)


# In[28]:


re.split('\W+',re_test_messy1)


# In[29]:


re.findall('\S+',re_test)


# In[30]:


re.findall('\S+',re_test_messy)


# In[31]:


re.findall('\w+',re_test_messy1)


# In[32]:


pep8_test = 'I try to follow PEP8 guidelines'
pep7_test = 'I try to follow PEP7 guidelines'
peep8_test = 'I try to follow PEEP8 guidelines'


# In[33]:


re.findall('[a-z]+',pep8_test)


# In[34]:


re.findall('[A-Z]+',pep8_test)


# In[35]:


re.findall('[A-Z]+[0-9]+',pep7_test)


# In[36]:


re.sub('[A-Z]+[0-9]+','PEP8',pep7_test)


# In[37]:


re.sub('[A-Z]+[0-9]+','PEP8',peep8_test)


# In[38]:


pd.set_option('display.max_colwidth',100)


# In[39]:


data = pd.read_csv('SMSSpamCollection.tsv',sep='\t',header=None)
data.columns = ['label','body_text']


# In[40]:


data.head()


# In[41]:


import string
string.punctuation


# In[42]:


def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct

data['body_text_clean'] = data['body_text'].apply(lambda x:remove_punct(x))


# In[43]:


data.head()


# In[44]:


import re

def tokenize(text):
    tokens = re.split('\W+',text)
    return tokens

data['body_text_clean'] = data['body_text_clean'].astype(str)

data['body_text_tokenized']= data['body_text_clean'].apply(lambda x:tokenize(x.lower()))
data.head()


# In[45]:


import nltk


# In[46]:


stopword = nltk.corpus.stopwords.words('english')


# In[47]:


print(stopword[::10])


# In[48]:


def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text

data['body_text_nostop'] = data['body_text_tokenized'].apply(lambda x:remove_stopwords(x))
data.head()


# In[49]:


import nltk

ps = nltk.PorterStemmer()


# In[50]:


dir(ps)


# In[51]:


print(ps.stem('grows'))
print(ps.stem('growing'))
print(ps.stem('grow'))


# In[52]:


print(ps.stem('run'))
print(ps.stem('running'))
print(ps.stem('runner'))


# In[53]:


import pandas as pd
import re
import string
pd.set_option('display.max_colwidth',100)


# In[54]:


stopwords = nltk.corpus.stopwords.words('english')

data = pd.read_csv("SMSSpamCollection.tsv", sep="\t", header=None)
data.columns = ['label','label text']


# In[55]:


data.head()


# In[56]:


def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation])
    tokens = re.split("\W+",text)
    text = [word for word in tokens if word not in stopwords]
    return text

data['body_text_nostop'] = data['label text'].apply(lambda x:clean_text(x.lower()))
data.head(6)


# In[57]:


def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text
data['body_text_stemmed']=data['body_text_nostop'].apply(lambda x:stemming(x))
data.head()


# In[58]:


import nltk

wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()


# In[59]:


dir(wn)


# In[60]:


print(ps.stem('meanness'))
print(ps.stem('meaning'))


# In[61]:


print(wn.lemmatize('meanness'))
print(wn.lemmatize('meaning'))


# In[62]:


print(wn.lemmatize('electrical'))
print(wn.lemmatize('electricity'))


# In[63]:


print(wn.lemmatize('goose'))
print(wn.lemmatize('geese'))


# In[64]:


stopwords = nltk.corpus.stopwords.words('english')

data = pd.read_csv("SMSSpamCollection.tsv", sep="\t", header=None)
data.columns = ['label','label text']


# In[65]:


def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation])
    tokens = re.split("\W+",text)
    text = [word for word in tokens if word not in stopwords]
    return text

data['body_text_nostop'] = data['label text'].apply(lambda x:clean_text(x.lower()))
data.head(6)


# In[66]:


def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text
data['body_text_lemmatized']=data['body_text_nostop'].apply(lambda x:lemmatizing(x))
data.head(10)


# In[67]:


import pandas as pd
import string
import re 
import nltk
pd.set_option('display.max_colwidth',100)


# In[68]:


stopwords = nltk.corpus.stopwords.words('english')


# In[69]:


ps = nltk.PorterStemmer()


# In[70]:


data = pd.read_csv('SMSSpamCollection.tsv',sep='\t',header=None)


# In[71]:


data.columns=['label','body_text']


# In[72]:


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+',text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text



# In[73]:


from sklearn.feature_extraction.text import CountVectorizer


# In[74]:


count_vect = CountVectorizer(analyzer=clean_text)
X_counts = count_vect.fit_transform(data['body_text'])
print(X_counts.shape)
print(count_vect.get_feature_names())


# In[75]:


data_sample = data[0:20]

count_vect_sample = CountVectorizer(analyzer = clean_text)
X_counts_sample = count_vect_sample.fit_transform(data_sample['body_text'])
print(X_counts_sample.shape)
print(count_vect_sample.get_feature_names())


# In[76]:


X_counts_sample


# In[77]:


X_counts_df_sample = pd.DataFrame(X_counts_sample.toarray())
X_counts_df_sample


# In[78]:


X_counts_df_sample.columns = count_vect_sample.get_feature_names()


# In[79]:


X_counts_df_sample


# In[80]:


import pandas as pd
import string
import re 
import nltk
pd.set_option('display.max_colwidth',100)
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
data = pd.read_csv('SMSSpamCollection.tsv',sep='\t',header=None)
data.columns=['label','body_text']


# In[81]:


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+',text)
    text = " ".join([ps.stem(word) for word in tokens if word not in stopwords])
    return text
data['cleaned_text'] = data['body_text'].apply(lambda x:clean_text(x))
data.head()


# In[82]:


from sklearn.feature_extraction.text import CountVectorizer


# In[83]:


ngram_vect = CountVectorizer(ngram_range=(2,2))
X_counts = ngram_vect.fit_transform(data['cleaned_text'])


# In[84]:


print(X_counts.shape)


# In[85]:


print(ngram_vect.get_feature_names())


# In[86]:


data_sample = data[0:20]
ngram_vect_sample = CountVectorizer(ngram_range=(2,2))
X_counts_sample = ngram_vect_sample.fit_transform(data_sample['cleaned_text'])
print(X_counts_sample.shape)


# In[87]:


print(ngram_vect_sample.get_feature_names())


# In[88]:


X_counts_sample_df = pd.DataFrame(X_counts_sample.toarray())


# In[89]:


X_counts_sample_df.columns = ngram_vect_sample.get_feature_names()


# In[90]:


X_counts_sample_df


# In[91]:


import pandas as pd
import string
import re 
import nltk
pd.set_option('display.max_colwidth',100)
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
data = pd.read_csv('SMSSpamCollection.tsv',sep='\t',header=None)
data.columns=['label','body_text']


# In[92]:


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+',text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text
data['cleaned_text'] = data['body_text'].apply(lambda x:clean_text(x))
data.head()


# In[93]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[94]:


tfid_vect = TfidfVectorizer(analyzer = clean_text)


# In[95]:


X_tfidf = tfid_vect.fit_transform(data['body_text'])
print(X_tfidf.shape)
print(tfid_vect.get_feature_names())


# In[96]:


data_sample = data[0:20]


# In[97]:


tfidf_vect_sample = TfidfVectorizer(analyzer= clean_text)
X_tfidf_sample = tfidf_vect_sample.fit_transform(data_sample['body_text'])
print(X_tfidf_sample.shape)
print(tfidf_vect_sample.get_feature_names())


# In[98]:


X_tfidf_sample_df = pd.DataFrame(X_tfidf_sample.toarray())
X_tfidf_sample_df.columns = tfidf_vect_sample.get_feature_names()


# In[99]:


X_tfidf_sample_df


# In[100]:


import pandas as pd
data = pd.read_csv("SMSSpamCollection.tsv",sep="\t",header=None)
data.columns=['label','body_text']


# In[101]:


data['body_len'] = data['body_text'].apply(lambda x:len(x)-x.count(" "))
data.head()


# In[102]:


import string

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text)-text.count(" ")),3)*100

data['punct%'] = data['body_text'].apply(lambda x:count_punct(x))
data.head()


# In[103]:


from matplotlib import pyplot
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[104]:


bins =np.linspace(0,200,40)

pyplot.hist(data[data['label']=='spam']['body_len'],bins,alpha=0.5,density=True,label='spam')
pyplot.hist(data[data['label']=='ham']['body_len'],bins,alpha=0.5,density=True,label='ham')
pyplot.legend(loc='upper left')
pyplot.show()


# In[105]:


bins =np.linspace(0,50,40)

pyplot.hist(data[data['label']=='spam']['punct%'],bins,alpha=0.5,density=True,label='spam')
pyplot.hist(data[data['label']=='ham']['punct%'],bins,alpha=0.5,density=True,label='ham')
pyplot.legend(loc='upper right')
pyplot.show()


# In[106]:


import pandas as pd

data = pd.read_csv("SMSSpamCollection.tsv",sep="\t",header=None)
data.columns=['label','body_text']


# In[107]:


import string 

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text)-text.count(" ")),3)*100

data['body_len'] = data['body_text'].apply(lambda x:len(x)-x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x:count_punct(x))
data.head()


# In[108]:


from matplotlib import pyplot
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[109]:


bins = np.linspace(0,200,40)
pyplot.hist(data['body_len'],bins)
pyplot.title("Body Length Distribution")
pyplot.show()


# In[110]:


bins = np.linspace(0,50,40)
pyplot.hist(data['punct%'],bins)
pyplot.title("Punctuation % Distribution")
pyplot.show()


# In[111]:


for i in [1,2,3,4,5]:
    pyplot.hist((data['punct%'])**(1/i),bins=40)
    pyplot.title("Transformation: 1/{}".format(str(i)))
    pyplot.show()


# In[112]:


import nltk
import pandas as pd
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
import string

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv("SMSSpamCollection.tsv",sep="\t",header=None)
data.columns = ['label','body_text']

def count_punct(text):
    count = sum([1 for word in text if word in string.punctuation])
    return round((count/len(text)-text.count(" ")),3)*100

data['body_len'] = data['body_text'].apply(lambda x:len(x)-x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x:count_punct(x))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split("\W+",text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

tfidf_vector = TfidfVectorizer(analyzer=clean_text)
x_tfidf = tfidf_vector.fit_transform(data['body_text'])


# In[113]:


X_features = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X_tfidf.toarray())],axis=1)
X_features.head()


# In[114]:


from sklearn.ensemble import RandomForestClassifier


# In[115]:


print(dir(RandomForestClassifier))
print(RandomForestClassifier())


# In[116]:


from sklearn.model_selection import KFold,cross_val_score
rf = RandomForestClassifier(n_jobs=-1)
k_fold = KFold(n_splits=5)
cross_val_score (rf,X_features,data['label'],cv=k_fold,scoring='accuracy',n_jobs=-1)


# In[117]:


import nltk
import pandas as pd
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
import string

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv("SMSSpamCollection.tsv",sep="\t",header=None)
data.columns = ['label','body_text']

def count_punct(text):
    count = sum([1 for word in text if word in string.punctuation])
    return round((count/len(text)-text.count(" ")),3)*100

data['body_len'] = data['body_text'].apply(lambda x:len(x)-x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x:count_punct(x))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split("\W+",text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

tfidf_vector = TfidfVectorizer(analyzer=clean_text)
x_tfidf = tfidf_vector.fit_transform(data['body_text'])

X_features = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X_tfidf.toarray())],axis=1)
X_features.head()


# In[118]:


from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X_features,data['label'],test_size=0.2)


# In[119]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50,max_depth=20,n_jobs=-1)
rf_model = rf.fit(X_train,Y_train)


# In[120]:


sorted(zip(rf_model.feature_importances_,X_train.columns),reverse=True)[0:10]


# In[121]:


Y_pred = rf_model.predict(X_test)
precision,recall,fscore,support = score(Y_test,Y_pred,pos_label='spam',average='binary')


# In[122]:


print('Precision: {}/Recall: {}/Accuracy: {}'.format(round(precision,3),round(recall,3),(Y_pred==Y_test).sum()/len(Y_pred)))


# In[123]:


import nltk
import pandas as pd
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
import string

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv("SMSSpamCollection.tsv",sep="\t",header=None)
data.columns = ['label','body_text']

def count_punct(text):
    count = sum([1 for word in text if word in string.punctuation])
    return round((count/len(text)-text.count(" ")),3)*100

data['body_len'] = data['body_text'].apply(lambda x:len(x)-x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x:count_punct(x))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split("\W+",text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

tfidf_vector = TfidfVectorizer(analyzer=clean_text)
x_tfidf = tfidf_vector.fit_transform(data['body_text'])

X_features = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X_tfidf.toarray())],axis=1)
X_features.head()


# In[124]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split


# In[125]:


X_train,X_test,Y_train,Y_test=train_test_split(X_features,data['label'],test_size=0.2)


# In[126]:


def train_RF(n_est,depth):
    rf = RandomForestClassifier(n_estimators=n_est,max_depth=depth,n_jobs=-1)
    rf_model = rf.fit(X_train,Y_train)
    Y_pred = rf_model.predict(X_test)
    precison,recall,fscore,support = score(Y_test,Y_pred,pos_label="spam",average="binary")
    print('Est:{}/Depth: {} --->Precision: {}/Recall: {},Accuracy: {}'.format(n_est,depth,round(precision,3),round(recall,3),(Y_pred==Y_test).sum()/len(Y_pred)))


# In[127]:


for n_est in [10,50,100]:
    for depth in [10,20,30,None]:
          train_RF(n_est,depth)


# In[128]:


import nltk
import pandas as pd
import re 
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import string

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv("SMSSpamCollection.tsv",sep="\t",header=None)
data.columns = ['label','body_text']

def count_punct(text):
    count = sum([1 for word in text if word in string.punctuation])
    return round((count/len(text)-text.count(" ")),3)*100

data['body_len'] = data['body_text'].apply(lambda x:len(x)-x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x:count_punct(x))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split("\W+",text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

tfidf_vector = TfidfVectorizer(analyzer=clean_text)
x_tfidf = tfidf_vector.fit_transform(data['body_text'])
X_tfidf_feat = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X_tfidf.toarray())],axis=1)

count_vect = CountVectorizer(analyzer=clean_text)
X_count = count_vect.fit_transform(data['body_text'])
X_count_feat = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X_count.toarray())],axis=1)

X_count_feat.head()


# In[129]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV



# In[130]:


rf = RandomForestClassifier()
param = {
    'n_estimators':[10,150,300],
    'max_depth':[10,60,90,None]
}

gs = GridSearchCV(rf,param,cv=5,n_jobs=-1)
gs_fit = gs.fit(X_tfidf_feat,data['label'])
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score',ascending=False)[0:5]


# In[132]:


rf = RandomForestClassifier()
param = {
    'n_estimators':[10,150,300],
    'max_depth':[10,60,90,None]
}

gs = GridSearchCV(rf,param,cv=5,n_jobs=-1)
gs_fit = gs.fit(X_count_feat,data['label'])
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score',ascending=False)[0:5]


# In[134]:


import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv("SMSSpamCollection.tsv",sep="\t",header=None)
data.columns = ['label','body_text']


# In[135]:


def count_punct(text):
    count = sum([1 for word in text if word in string.punctuation])
    return round(count/(len(text)-text.count(" ")),3)*100

data['body_len'] = data['body_text'].apply(lambda x:len(x)-x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x:count_punct(x))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split("\W+",text)
    text = [ps.stem(word) for word in tokens if word not in stopwords ]
    return text

tfidf_vector = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vector.fit_transform(data['body_text'])

X_features = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X_tfidf.toarray())],axis=1)
X_features.head()


# In[137]:


from sklearn.ensemble import GradientBoostingClassifier


# In[143]:


print(dir(GradientBoostingClassifier))
print(GradientBoostingClassifier().get_params())


# In[145]:


from sklearn.metrics import  precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test =  train_test_split(X_features,data['label'],test_size=0.2)


# In[156]:


def train_GB(est,max_depth,lr):
    gb = GradientBoostingClassifier(n_estimators=est,max_depth=max_depth,learning_rate=lr)
    gb_model=gb.fit(X_train,Y_train)
    Y_pred = gb_model.predict(X_test)
    precision,recall,fscore,support = score(Y_test,Y_pred,pos_label='spam',average='binary')
    print("Est: {}/Depth: {}---->Precision: {}/Recall: {}/Accuracy: {}".format(est,max_depth,round(precision,3),round(recall,3),(Y_pred==Y_test).sum()/len(Y_pred)))


# In[157]:


for n_est in [50,100,150]:
    for max_depth in [3,7,11,15]:
        for lr in [0.01,0.1,1]:
            train_GB(n_est,max_depth,lr)


# In[158]:


import nltk
import pandas as pd
import re 
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import string

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv("SMSSpamCollection.tsv",sep="\t",header=None)
data.columns = ['label','body_text']

def count_punct(text):
    count = sum([1 for word in text if word in string.punctuation])
    return round(count/(len(text)-text.count(" ")),3)*100

data['body_len'] = data['body_text'].apply(lambda x:len(x)-x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x:count_punct(x))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split("\W+",text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

tfidf_vector = TfidfVectorizer(analyzer=clean_text)
x_tfidf = tfidf_vector.fit_transform(data['body_text'])
X_tfidf_feat = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X_tfidf.toarray())],axis=1)

count_vect = CountVectorizer(analyzer=clean_text)
X_count = count_vect.fit_transform(data['body_text'])
X_count_feat = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X_count.toarray())],axis=1)

X_count_feat.head()


# In[159]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


gb = GradientBoostingClassifier()
param = {
    'n_estimators':[100,150],
    'max_depth':[7,11,15],
    'learning_rate':[0.01,0.1,1]
}

gs = GridSearchCV(gb,param,cv=5,n_jobs=-1)
gs_fit = gs.fit(X_count_feat,data['label'])
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score',ascending=False)[0:5]


# In[161]:


import nltk
import pandas as pd
import re 
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import string

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv("SMSSpamCollection.tsv",sep="\t",header=None)
data.columns = ['label','body_text']

def count_punct(text):
    count = sum([1 for word in text if word in string.punctuation])
    return round(count/(len(text)-text.count(" ")),3)*100

data['body_len'] = data['body_text'].apply(lambda x:len(x)-x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x:count_punct(x))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split("\W+",text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text


# In[164]:


from sklearn.model_selection import train_test_split


# In[165]:


X_train,X_test,Y_train,Y_test = train_test_split(data[['body_text','body_len','punct%']],data['label'],test_size=0.2)


# In[169]:


tfidf_vect= TfidfVectorizer(analyzer=clean_text)
tfidf_vect_fit= tfidf_vect.fit(X_train['body_text'])

tfidf_train = tfidf_vect_fit.transform(X_train['body_text'])
tfidf_test = tfidf_vect_fit.transform(X_test['body_text'])

X_train_vect = pd.concat([X_train[['body_len','punct%']].reset_index(drop=True),pd.DataFrame(tfidf_train.toarray())],axis=1)
X_test_vect = pd.concat([X_test[['body_len','punct%']].reset_index(drop=True),pd.DataFrame(tfidf_test.toarray())],axis=1)

X_train_vect.head()


# In[173]:


from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
import time


rf = RandomForestClassifier(n_estimators=150 , max_depth=None, n_jobs=-1)
start = time.time()
rf_model = rf.fit(X_train_vect,Y_train)
end = time.time()
fit_time = (end-start)
start = time.time()
Y_pred = rf_model.predict(X_test_vect)
end = time.time()

pred_time = (end - start)

precision,recall,fscore,support = score(Y_test,Y_pred,pos_label='spam',average='binary')
print("Fit time: {}/Predict time:{} ---->Precision: {}/Recall: {}/Accuracy: {}".format(round(fit_time,3),round(pred_time,3),round(precision,3),round(recall,3),(Y_pred==Y_test).sum()/len(Y_pred)))


# In[176]:


gb = GradientBoostingClassifier(n_estimators=150 , max_depth=11)
start = time.time()
gb_model = gb.fit(X_train_vect,Y_train)
end = time.time()
fit_time = (end-start)
start = time.time()
Y_pred = gb_model.predict(X_test_vect)
end = time.time()

pred_time = (end - start)

precision,recall,fscore,support = score(Y_test,Y_pred,pos_label='spam',average='binary')
print("Fit time: {}/Predict time:{} ---->Precision: {}/Recall: {}/Accuracy: {}".format(round(fit_time,3),round(pred_time,3),round(precision,3),round(recall,3),(Y_pred==Y_test).sum()/len(Y_pred)))


# In[ ]:




