# Wikipedia Summarization for Unseen Object Recognition
Tensorflow implementation of zero-shot learning through Wikipedia textual description using using [seq2seq library](https://www.tensorflow.org/api_guides/python/contrib.seq2seq)
The text summarization model is based on [dongjun-Lee/text-summarization-tensorflow](https://github.com/dongjun-Lee/text-summarization-tensorflow)

## Model
we are  adopting  the  Encoder-Decoder model for text summarization. The summarization model learns to pick the words that are the most important in the article. We then take the average word embedding of these important words as our textual feature of this class, and train our zero-shot image classifier using it.

## Word Embedding
Used [Glove pre-trained vectors](https://nlp.stanford.edu/projects/glove/) to initialize word embedding.

## Requirements
- Python 3
- Tensorflow (>=1.8.0)
- pip install -r requirements.txt

## Download data and model

### Download visual training data
The Animal with Attribute dataset is preprocessing using ResNet and stored into TFRecord format for high-capacity data loading in tensorflow. To run the model,  download [TFRecord.zip](https://drive.google.com/file/d/1hoTci2t50cwQme7oJs-lDh3UGHzVKRlU/view?usp=sharing), and locate it in the project root directory. Then,
```
$ unzip TFRecord.zip
```
Original data can be download here [AWA2-data.zip](https://cvml.ist.ac.at/AwA2/AwA2-data.zip)


### Download pretrained seq2seq model 
To use pre-trained seq2seq model (this is trained on the task of text summarization without zero-shot learning), download [pre_trained.zip](https://drive.google.com/file/d/1V8pS1eoiv51wfiVp2rOB7IvJ5PeQs2n-/view?usp=sharing), and locate it in the project root directory. Then,
```
$ unzip pre_trained.zip
```

## Train
To train the model, visual data and pretrained seq2seq must be downloaded and located to the project root directory. Our model setting is in the file ```global_setting_AWA2.py``` which stored the learning rate, number of iteration, data path, and batch size. The command in this section would reproduce the result in our report up to a stochastic noise (since we randomly initialize all models).

### Run baseline
To train baseline model, run the command
```
$ python ./baselines_zs_AWA2.py
```
Notice that in ```baselines_zs_AWA2.py``` line 30, you can replace ```path_w2v = './data/glove_vecs.npy'``` with any embedding in ```./data/```

### Run pretrained summarization zero-shot learning
To train zero-shot learning model while fixing the seq2seq model to its pretrained value, run the command
```
$ python ./pretrain_summarization_zs.py
```
Notice that the model detail to use GPU 0 to train this model. It is recommended that the GPU has at least 8GB of memory. The output result is stored in ```./result```.

### Run fine-tune summarization zero-shot learning
To train zero-shot learning model and fine-tuning the seq2seq model, run the command
```
$ python ./finetune_summarization_zs.py
```
The output result is stored in ```./result```.

### Get wiki data (Optional)
We have include the extracted wikipedia in ```./wiki_data/wiki_article.pkl```. To re-extract the wikipedia, run the command
```
$ python ./get_wiki_text.py
```

## Sample input
Sample input can be found in ```./sample_input/```
