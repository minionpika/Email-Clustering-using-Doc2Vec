import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import os, sys, email
import gensim
from gensim.models import Doc2Vec
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from string import punctuation
import timeit
from sklearn.cluster import KMeans
from sklearn import metrics
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math


def get_text_from_email(msg):
    # To get the content from email objects

    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append(part.get_payload())
    return ''.join(parts)


def split_email_addresses(line):
    # To separate multiple email addresses

    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x: x.strip(), addrs))
    else:
        addrs = None
    return addrs


def email_cleaning(text):
    # Data cleaning

    email = text.lower()
    # clean and tokenize document string
    email_content = email.split()
    word_list = []
    for i in email_content:
        x = 0
        if (('http' not in i) and ('@' not in i) and ('<.*?>' not in i) and i.isalnum() and (not i in stop_words)):
            word_list += [i]

    return word_list


def preprocessing(text):

    # Data Pre-processing
    # remove numbers
    number_tokens = [re.sub(r'[\d]', ' ', i) for i in text]
    number_tokens = ' '.join(number_tokens).split()
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
    # remove empty
    length_tokens = [i for i in stemmed_tokens if len(i) > 1]
    return length_tokens


# ---------------------------------------------------------------- main
print('1')

start = timeit.default_timer()

# ---------------------------------------------------------------- small data

emails_df = pd.read_csv('emails.csv').head(20000)
emails_df2 = pd.read_csv('emails_2only.csv')
emails_df = pd.concat([emails_df, emails_df2])
emails_df.index = [i for i in range(20002)]

# ------------------------------------------------------------ Data Preparation

messages = list(map(email.message_from_string, emails_df['message']))
emails_df.drop('message', axis=1, inplace=True)
keys = messages[0].keys()
for key in keys:
    emails_df[key] = [doc[key] for doc in messages]

# ------------------------------------------------------------ Parse content from emails

emails_df['content'] = list(map(get_text_from_email, messages))

# ------------------------------------------------------------ Split multiple email addresses

emails_df['From'] = emails_df['From'].map(split_email_addresses)
emails_df['To'] = emails_df['To'].map(split_email_addresses)

# ------------------------------------------------------------ Extract the root of 'file' as 'user'

print('2')
emails_df['user'] = emails_df['file'].map(lambda x: x.split('/')[0])
del messages
emails_df['Subjcontent'] = emails_df['Subject'] + " " + emails_df['content']


# ------------------------------------------------------------ Create list of tagged emails

print('3')
LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
all_content = []
texts = []
j = 0
k = 0
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
p_stemmer = PorterStemmer()

for em in emails_df.Subjcontent:
    # Data cleaning
    clean_content = email_cleaning(em)
    # Pre-processing
    processed_email = preprocessing(clean_content)
    # add tokens to list
    if processed_email:
        all_content.append(LabeledSentence1(processed_email, [j]))
        j += 1
    k += 1

print("Number of emails processed: ", k)
print("Number of non-empty emails vectors: ", j)

# ------------------------------------------------------------ Doc 2 Vec
print('4')

d2v_model = Doc2Vec(all_content, vector_size=300, window=10, min_count=300, workers=7, dm=1,
                    alpha=0.025, min_alpha=0.001)
print('4-5')

d2v_model.train(all_content, total_examples=d2v_model.corpus_count, epochs=10, start_alpha=0.002, end_alpha=-0.016)


print(d2v_model.docvecs.most_similar(1))
print('5')

# ------------------------------------------------------------ K-means on the model


hi = 10
lo = 5
mid = 0
opt_k = 5
thold = 0.1
mxiter = 150
while lo <= hi:
    mid = (hi + lo) // 2
    kmeans_model = KMeans(n_clusters=mid, init='k-means++', max_iter=mxiter)
    X = kmeans_model.fit(d2v_model.docvecs.doctag_syn0)
    f1 = kmeans_model.inertia_
    kmeans_model = KMeans(n_clusters=mid + 1, init='k-means++', max_iter=mxiter)
    X = kmeans_model.fit(d2v_model.docvecs.doctag_syn0)
    f2 = kmeans_model.inertia_
    dl = f1 - f2
    print('-------------', f1, f2, '-------------')
    if dl < thold:
        hi = mid - 1
        opt_k = mid
    else:
        lo = mid + 1

fit_array = np.zeros(shape=d2v_model.docvecs.doctag_syn0.shape)

'''for i in range(len(fit_array)):
    fit_array[i]=np.divide(d2v_model.docvecs.doctag_syn0[i] , np.sqrt(np.dot(d2v_model.docvecs.doctag_syn0[i], d2v_model.docvecs.doctag_syn0[i])))'''

fit_array = d2v_model.docvecs.doctag_syn0
np.savetxt('doc2vecOutput.csv', X=np.asarray(fit_array), delimiter=',')

print(fit_array.shape)

kmeans_model = KMeans(n_clusters=opt_k, init='k-means++', max_iter=mxiter)
X = kmeans_model.fit(fit_array)
labels = kmeans_model.labels_.tolist()

print('6')

l = kmeans_model.fit_predict(fit_array)
pca = PCA(n_components=2).fit(fit_array)
data_point = pca.transform(fit_array)
print('7')

# ------------------------------------------------------------ plot cluster

plt.figure()

label1 = ["#FFA500", "#FFFF00", "#A9A9A9", "#FF0000", "#778899", "#0D98BA", "#8B008B", "#4B0082", "#FF00FF", "#E6E6FA"]

color = [label1[i] for i in labels]

color[len(color) - 1] = "#7CFC00"
color[len(color) - 2] = "#00FF00"

plt.scatter(data_point[:, 0], data_point[:, 1], c=color)

print('8')

centroids = kmeans_model.cluster_centers_
centroid_point = pca.transform(centroids)
plt.show()

# ----------------------------------------------------------- execution time
print('9')

stop = timeit.default_timer()
execution_time = stop - start

print(execution_time, ' sec.')