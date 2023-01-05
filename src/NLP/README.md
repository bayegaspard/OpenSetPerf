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
        
Output: Web based state management attacks | Confidence: 0.8453215545350422
Output: Linux based website vulnerability scanner | Confidence: 0.8388613892457681
Output: algorithms preventing basic hacking tools | Confidence: 0.8348181744196423
Output: dictionary password cracking lists available | Confidence: 0.8337179010200412
Output: control enterprise security software solutions | Confidence: 0.8326010253664153
Output: DBA database security involves Fine | Confidence: 0.8312824686836693
Output: exist password cracking tools based | Confidence: 0.8287541910193648
Output: cially supported vulnerability checks updated | Confidence: 0.8269457315316675
Output: deployed large applications database servers | Confidence: 0.8268889482931712
Output: powerful password cracking tool available | Confidence: 0.826203337858431
```

## Usage: Packet Data
```python
# Get PCAP data as Hexadecimal or ASCII (as a list)
# Example of getting packet information as ASCII
>>> import pandas as pd
>>> import numpy as np

# Getting UNSW-NB15 dataset
>>> !gdown "https://drive.google.com/uc?id=1rRLWQbfxxynlfKyW4OIi1n6gRynufspj&confirm=t"
>>> df = pd.read_csv('Payload_data_UNSW.csv', nrows=10000)

# Get Payload bytes of a random packet (1500 bytes)
>>> ascii = df2.iloc[2457,:1500].values

# Get words from the ASCII values
>>> pcap_words = set(''.join([chr(i) if (i in range(65,91) or i in range(97,123)) else ' ' for i in ascii]).split())
>>> bert_words = set(similar.bert_words)
>>> input_words = pcap_words & bert_words

# Find similay words
>>> similar_dict = similar.find_similar_words(
    input_context = '',
    input_words=list(input_words),
    output_words_ngram=0,
    max_output_words=10,
    context_similarity_factor = 0.25,
    output_filter_factor = 0.5,
    single_word_split=True,
    uncased_lemmatization=True
    )
>>> for k,v in similar_dict.items():
        print(f'Output: {k:16} | Confidence: {v}')
        
Output: Session hijacking tools monitor active communication sessions | Confidence: 0.8720203683285169
Output: features include capturing packets importing pcap files displaying protocol information | Confidence: 0.8688546718391735
Output: underlying Directory Informa tion Base | Confidence: 0.8685651149910566
Output: incorporates local database authentication gaining administrative level access | Confidence: 0.8663377026638005
Output: collectively address network layer encryp tion entity authentication | Confidence: 0.8652275706437064
Output: MAC flooding attacks Leverage port level security | Confidence: 0.8634708796414688
Output: manipulating source logging facilities | Confidence: 0.86337432093541
Output: version applications running location access control permissions intrusion detection tests | Confidence: 0.8633725860195026
Output: Utmp facility logs username line | Confidence: 0.86322374528859
Output: basic features include capturing packets importing pcap files displaying protocol | Confidence: 0.8629671383412492
```
