# Find Similar Words using BERT

## Install the Package

```
pip install BERTSimilarWords
```

## Import the Module

```python
>>> from BERTSimilarWords import BERTSimilarWords
```

## Get Vocabulary from two Books

```python
# Book 1: Cybersecurity – Attack and Defense Strategies, Yuri Diogenes, Erdal Ozkaya
# https://docs.google.com/document/d/10K2y7Kh_yJ9vs10HSnorfcq05dNYbeBD/edit
>>> !gdown "https://drive.google.com/uc?id=10K2y7Kh_yJ9vs10HSnorfcq05dNYbeBD&confirm=t"

# Book 2: The Hacker’s Handbook, Susan Young, Dave Aitel
# https://docs.google.com/document/d/10lgc_F3Y1SHKMSiEXbeoOM0LyKoyN34t/edit
>>> !gdown "https://drive.google.com/uc?id=10lgc_F3Y1SHKMSiEXbeoOM0LyKoyN34t&confirm=t"

>>> similar = BERTSimilarWords().load_dataset(dataset_path=['Book 1.docx','Book 2.docx'])
```

## Usage

```python
>>> similar_dict = similar.find_similar_words(
    input_context = '',
    input_words=['database', 'software', 'attack'],
    output_words_ngram=5,
    max_output_words=10,
    context_similarity_factor = 0.25,
    output_filter_factor = 0.5,
    single_word_split=True,
    uncased_lemmatization=True
    )
>>> for k,v in similar_dict.items():
      print(f'Output: {k:16} | Confidence: {v}')
```
