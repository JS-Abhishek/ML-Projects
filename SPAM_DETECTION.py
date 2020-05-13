#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tarfile
import urllib


# In[2]:


DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets","spam")

def fetch_spam_data(spam_url = SPAM_URL, spam_path = SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename,url in (("ham.tar.bz2",HAM_URL),("spam.tar.bz2",SPAM_URL)):
        path = os.path.join(spam_path,filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url,path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=SPAM_PATH)
        tar_bz2_file.close()
            


# In[3]:


fetch_spam_data()


# In[4]:


HAM_DIR = os.path.join(SPAM_PATH,"easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH,"spam")
ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name)>20]
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name)>20]


# In[5]:


HAM_DIR


# In[6]:


import email
import email.policy

def load_emails(is_spam,filename,spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path,directory,filename),"rb") as file:
        return email.parser.BytesParser(policy=email.policy.default).parse(file)


# In[7]:


ham_emails = [load_emails(is_spam=False,filename=name) for name in ham_filenames]
spam_emails =[load_emails(is_spam=True,filename=name) for name in spam_filenames]


# In[8]:


print(spam_emails[6].get_content().strip())


# In[9]:


def get_email_structure(email):
    if isinstance(email,str):
        return email
    payload = email.get_payload()
    if isinstance(payload,list):
        return "multipart({})".format(",".join([get_email_structure(sub_email) for sub_email in payload]))
    else:
        return email.get_content_type()


# In[10]:


from collections import Counter

def structure_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] +=1
    return structures


# In[11]:


structure_counter(ham_emails).most_common()


# In[12]:


structure_counter(spam_emails).most_common()


# In[13]:


for header,values in ham_emails[0].items():
    print(header,':',values)


# In[14]:


spam_emails[0]["Subject"]


# In[15]:


import numpy as np
from sklearn.model_selection import train_test_split
X = np.array(ham_emails + spam_emails)
Y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))


# In[16]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


# In[17]:


len(Y_train)


# In[18]:


import re
from html import unescape

def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>','',html,flags=re.M|re.S|re.I)
    text = re.sub('<a\s.*?>','HYPERLINK',text,flags=re.M|re.S|re.I)
    text = re.sub('<.*?>','',text,flags=re.M|re.S|re.I)
    text = re.sub(r'(\s*\n)+','\n',text,flags=re.M|re.S|re.I)
    return unescape(text)


# In[19]:


html_spam_email = [email for email in X_train[Y_train==1] 
                    if get_email_structure(email)=='text/html']
sample_email = html_spam_email[7]
print(sample_email.get_content().strip()[:1000],"...")


# In[20]:


print(html_to_plain_text(sample_email.get_content())[:1000],'...')


# In[21]:


def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain","text/html"):
            continue
        try:
            content = part.get_content()
        except:
            content = str(part.get_payload())
        if ctype=="text/plain":
            return content
        else:
            html = content
        if html:
            return html_to_plain_text(html)


# In[22]:


print(email_to_text(sample_email)[:100],'...')


# In[23]:


try:
    import nltk
    
    stemmer = nltk.PorterStemmer()
    for word in ("Computations","Computation","Computing","Computed","Compute","Compulsive"):
        print(word,"=>",stemmer.stem(word))
except ImportError:
    print("Error: Stemming requires NLTK module")
    stemmer = None


# In[24]:


try:
    import urlextract
    
    url_extractor = urlextract.URLExtract()
    print(url_extractor.find_urls("will it detect any url from text like google.com"))
except ImportError:
    print("Error: Replacing URLs require urlextract module")
    url_extractor = None


# In[25]:


from sklearn.base import BaseEstimator, TransformerMixin

class Emailtoword_trans(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True,lower_case=True,
                 remove_punctuation=True,replace_urls=True,replace_numbers=True,stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self,X,Y=None):
        return self
    def transform(self,X,Y=None):
        X_transformed=[]
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url:len(url),reverse = True)
                for url in urls:
                    text = text.replace(url," URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?','NUMBER',text)
            if self.remove_punctuation:
                text = re.sub(r'\W+',' ',text,flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word,count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word]+=count
                word_counts = stemmed_word_counts
                X_transformed.append(word_counts)
        return np.array(X_transformed)
                


# In[26]:


X_few = X_train[:3]
X_few_wordcounts = Emailtoword_trans().fit_transform(X_few)
X_few_wordcounts


# In[72]:


from scipy.sparse import csr_matrix

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.most_common_ = most_common
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        #print(self.vocabulary_)
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
                #print(self.vocabulary_.get(word,0))
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))


# In[73]:


vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
X_few_vectors


# In[74]:


X_few_vectors.toarray()


# In[75]:


vocab_transformer.vocabulary_


# In[76]:


from sklearn.pipeline import Pipeline

preprocess_pipeline = Pipeline([
    ("email_to_wordcount",Emailtoword_trans()),
    ("wordcount_to_vector",WordCounterToVectorTransformer())
])

X_train_transformed = preprocess_pipeline.fit_transform(X_train)


# In[77]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log_clf = LogisticRegression(solver = "lbfgs", random_state = 42)
score = cross_val_score(log_clf,X_train_transformed,Y_train,cv=3,verbose = 3)
score.mean()


# In[78]:


from sklearn.metrics import precision_score, recall_score
X_test_transformed = preprocess_pipeline.transform(X_test)

log_clf.fit(X_train_transformed,Y_train)
Y_pred = log_clf.predict(X_test_transformed)

precision = precision_score(Y_test,Y_pred)
recall = recall_score(Y_test,Y_pred)


# In[80]:


print(precision*100)
print(recall*100)

