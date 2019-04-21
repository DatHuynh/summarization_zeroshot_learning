from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import collections
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import pdb
nltk.download('stopwords')

train_article_path = "sumdata/train/train.article.txt"
train_title_path = "sumdata/train/train.title.txt"
valid_article_path = "sumdata/train/valid.article.filter.txt"
valid_title_path = "sumdata/train/valid.title.filter.txt"

def evaluate_macro(idx_preds_test_v,labels_tst):
    count = 0.0
    for idx,l in enumerate(labels_tst):
        if l in idx_preds_test_v[idx]:
            count+=1.0
    return count /len(labels_tst)

def evaluate_micro(idx_preds_test_v,labels_tst,seen_classes,unseen_classes):
    classes_in_tst = np.unique(labels_tst)
    acc_per_class=np.zeros(len(seen_classes)+len(unseen_classes))
    indicator = np.zeros(idx_preds_test_v.shape[0])
    
    for idx,l in enumerate(labels_tst):
        if l in idx_preds_test_v[idx]:
            indicator[idx]=1.0
            
    for idx_c,c in enumerate(classes_in_tst):
        mask = labels_tst==c
        if np.sum(mask) == 0:
            acc_per_class[idx_c] = -1
        else:
            acc_per_class[idx_c]=np.mean(indicator[mask])
            
    mask_all = acc_per_class != -1
    acc_all = np.mean(acc_per_class[mask_all])
    
    mask_seen = acc_per_class[seen_classes] != -1
    acc_seen = np.mean(acc_per_class[seen_classes][mask_seen])
    
    mask_unseen = acc_per_class[unseen_classes] != -1
    acc_unseen = np.mean(acc_per_class[unseen_classes][mask_unseen])
    
    return acc_all,acc_seen,acc_unseen,acc_per_class

def clean_str(sentence):
    sentence = re.sub("[#.]+", "#", sentence)
    return sentence


def get_text_list(data_path, toy):
    with open (data_path, "r", encoding="utf-8") as f:
        if not toy:
            return [clean_str(x.strip()) for x in f.readlines()]
        else:
            return [clean_str(x.strip()) for x in f.readlines()][:50000]


def build_dict(step, toy=False):
    if step == "train":
        train_article_list = get_text_list(train_article_path, toy)
        train_title_list = get_text_list(train_title_path, toy)

        words = list()
        for sentence in train_article_list + train_title_list:
            for word in word_tokenize(sentence):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    elif step == "valid":
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

    article_max_len = 50
    summary_max_len = 15

    return word_dict, reversed_dict, article_max_len, summary_max_len

def build_wiki_text(wiki_data, word_dict, article_max_len):
    x = [word_tokenize(d.lower()) for d in wiki_data]
    ## filter out stopwords ##
    x = [[w for w in d if w not in stopwords.words('english')] for d in x]
    ## filter out stopwords ##
    x = [[word_dict.get(w) for w in d if w in word_dict] for d in x]
    x = [d[:article_max_len] for d in x]
    x = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in x]
    
    return x

def decode_id(wiki_context,reversed_dict):
    string_context = []
    assert isinstance(wiki_context,list)
    for context in wiki_context:
        text = []
        for word_id in context:
            text.append(reversed_dict[word_id])
        string=' '.join(text) 
        string_context.append(string)
    return string_context

def build_dataset(step, word_dict, article_max_len, summary_max_len, toy=False):
    if step == "train":
        article_list = get_text_list(train_article_path, toy)
        title_list = get_text_list(train_title_path, toy)
    elif step == "valid":
        article_list = get_text_list(valid_article_path, toy)
    else:
        raise NotImplementedError

    x = [word_tokenize(d) for d in article_list]
    x = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in x]
    x = [d[:article_max_len] for d in x]
    x = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in x]
    
    if step == "valid":
        return x
    else:        
        y = [word_tokenize(d) for d in title_list]
        y = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in y]
        y = [d[:(summary_max_len - 1)] for d in y]
        return x, y


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


def get_init_embedding(reversed_dict, embedding_size):
    glove_file = "glove/glove.42B.300d.txt"
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    print("Loading Glove vectors...")
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    # Assign random vector to <s>, </s> token
    word_vec_list[2] = np.random.normal(0, 1, embedding_size)
    word_vec_list[3] = np.random.normal(0, 1, embedding_size)

    return np.array(word_vec_list)
