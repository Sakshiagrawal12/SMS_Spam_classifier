#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('spam.csv',encoding="latin-1")
df


# In[3]:


df.sample(5)


# In[4]:


df.shape


# In[5]:


#1.Data cleaning
#2.EDA
#3.Text Preprocessing
#4.Model building
#5.Evaluation
#6.Improvement
#7.Website
#8.Deploy


# ## 1.Data Cleaning

# In[6]:


df.info()


# In[7]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[8]:


df


# In[9]:


df.rename(columns={'v1':'target','v2':'text'},inplace=True)


# In[10]:


df


# In[11]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[12]:


df['target']=le.fit_transform(df['target'])


# In[13]:


df


# In[14]:


df.isnull().sum()


# In[15]:


df.duplicated().sum()


# In[16]:


df=df.drop_duplicates(keep='first')#keep=first is use to  keep first occurence and delete othe duplicate


# In[17]:


df.head()


# In[18]:


df.duplicated().sum()


# In[19]:


df.shape


# ## 2.EDA
# 

# In[20]:


import matplotlib.pyplot as plt
df['target'].value_counts()


# In[21]:


plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[22]:


#data is imbalance
import nltk


# In[23]:


nltk.download('punkt')
nltk.download('punkt_tab')


# In[24]:


df=df.copy()
df['num_character']=df['text'].apply(len)


# In[25]:


#number of word
df['words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[26]:


df.head()


# In[27]:


df.head()


# In[28]:


df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[29]:


df.head()


# In[30]:


df[df['target']==0][['num_character','words','num_sentences']].describe()


# In[31]:


df[df['target']==1][['num_character','words','num_sentences']].describe()


# In[32]:


import seaborn as sns


# In[33]:


sns.histplot(df[df['target']==0]['num_character'])
sns.histplot(df[df['target']==1]['num_character'],color='red')


# In[34]:


sns.histplot(df[df['target']==0]['words'])
sns.histplot(df[df['target']==1]['words'],color='red')


# In[35]:


sns.pairplot(df,hue='target')


# In[36]:


numeric_df=df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(),annot=True)


# ## 3.Data Preprocessing

#  - Lowercase
#  - Tokenization
#  - Removing special Characters
#  - Removing stop words and punctuation
#  - Stemming

# In[37]:


import nltk 
nltk.download('stopwords')


# In[38]:


import string
from nltk.corpus import stopwords
string.punctuation


# In[39]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[40]:


stopwords.words('english')


# In[41]:


def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    y = []
    for i in tokens:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# In[42]:


transform_text('I loved the Youtube lectures on Machine Learning!! How about You????')


# In[43]:


df['text'][2000]


# In[44]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem('loving')


# In[45]:


df['transformed_text']=df['text'].apply(transform_text)


# In[46]:


df.head()


# In[47]:


get_ipython().system('pip install wordcloud')


# In[48]:


from wordcloud import WordCloud
wc=WordCloud(width=700,height=700,min_font_size=10,background_color='white')


# In[49]:


spam_wc=wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))


# In[50]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[51]:


ham_wc=wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))


# In[52]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[53]:


for msg in df[df['target']==1]['transformed_text'].tolist():
    pass
    #print(msg)


# In[54]:


spam_corpus=[]
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[55]:


len(spam_corpus)


# In[56]:


from collections import Counter
#Counter(spam_corpus)


# In[57]:


Counter(spam_corpus).most_common(30)


# In[58]:


pd.DataFrame(Counter(spam_corpus).most_common(30))


# In[59]:


from collections import Counter
sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0],y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[60]:


ham_corpus=[]
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[61]:


pd.DataFrame(Counter(ham_corpus).most_common(30))


# In[62]:


len(ham_corpus)


# In[63]:


from collections import Counter
sns.barplot(x=pd.DataFrame(Counter(ham_corpus).most_common(30))[0],y=pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# ## 4. Model Building

# In[64]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
tfidf=TfidfVectorizer(max_features=3000)


# In[65]:


x=tfidf.fit_transform(df['transformed_text']).toarray()


# In[66]:


x


# In[67]:


x.shape


# In[68]:


y=df['target'].values


# In[69]:


y


# In[70]:


y.shape


# In[71]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[72]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score


# In[73]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[74]:


gnb.fit(x_train,y_train)
y_pred1=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(classification_report(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[75]:


mnb.fit(x_train,y_train)
y_pred2=mnb.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print(classification_report(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[76]:


bnb.fit(x_train,y_train)
y_pred3=bnb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(classification_report(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[77]:


get_ipython().system('pip install xgboost')


# In[78]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[79]:


svc=SVC(kernel='sigmoid',gamma=1.0)
knc=KNeighborsClassifier()
mnb=MultinomialNB()
dtc=DecisionTreeClassifier(max_depth=5)
lrc=LogisticRegression(solver='liblinear',penalty='l1')
rfc=RandomForestClassifier(n_estimators=50,random_state=2)
abc=AdaBoostClassifier(n_estimators=50,random_state=2)
bc=BaggingClassifier(n_estimators=50,random_state=2)
etc=ExtraTreesClassifier(n_estimators=50,random_state=2)
gbdt=GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb=XGBClassifier(n_estimators=50,random_state=2)


# In[80]:


clfs={
    'SVC':svc,
    'KN':knc,
    'NB':mnb,
    'DT':dtc,
    'LR':lrc,
    'RF':rfc,
    'AdaBoost':abc,
    'Bgc':bc,
    'ETC':etc,
    'GBDT':gbdt,
    'xgb':xgb
}


# In[81]:


from sklearn.metrics import accuracy_score , precision_score
def train_classifier(clf,x_train,y_train,x_test,y_test):
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)

    return accuracy,precision


# In[ ]:





# In[82]:


accuracy_scores=[]
precision_scores=[]

for name,clf in clfs.items():
    current_accuracy,current_precision=train_classifier(clf,x_train,y_train,x_test,y_test)

    print("For",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[83]:


performance1=pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Accuracy',ascending=False)


# In[84]:


performance1


# In[85]:


performance_df2=pd.melt(performance1,id_vars="Algorithm")


# In[86]:


sns.catplot(x='Algorithm',y='value',hue='variable',data=performance_df2,kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[87]:


performance=pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[88]:


performance


# In[89]:


performance_df1=pd.melt(performance,id_vars="Algorithm")


# In[90]:


sns.catplot(x='Algorithm',y='value',hue='variable',data=performance_df1,kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:





# In[ ]:




