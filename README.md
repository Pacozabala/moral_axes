# Undergraduate Thesis Preamble

This is to track changes and acknowledge the prior work done in this repository. The code and documentation is based on the original Mformer model, which can be found [here](https://github.com/joshnguyen99/moral_axes).

## Usage Guidelines

Please use a Python 3.11 environment to run the modules. All required libraries can be found in the `requirements.txt` file.

## Modifications to original repository

This section documents the changes done to the files and code from the original repository.

| Filename    | Changes Done |
| -------- | ------- |
| `requirements.txt` | Changed pip module versions to work with Python 3.11 |
| `.gitignore` | Commented out 'data' to allow the dataset and lexicon folder to be included in the remote repository. |
| `scripts/mfd/data/lexicons` | Created the `data` and `data/lexicons` folders to house downloaded dataset files. |

## Data and External Resources

This project makes use of several Moral Foundations Theory (MFT) lexicons and annotated datasets developed by prior research in moral psychology and computational social science.

### Moral Foundations Lexicons

The following lexicons are used in this work:

- **Moral Foundations Dictionary (MFD)**  
  Originally introduced by Graham et al. and made publicly available via the Moral Foundations website and OSF.

- **Moral Foundations Dictionary 2.0 (MFD 2.0)**  
  An updated version of the original MFD, distributed via the Open Science Framework (OSF).

- **Extended Moral Foundations Dictionary (eMFD)**  
  A probabilistic extension of the MFD introduced by Hopp et al.

The **eMFD** resource is distributed under the GNU GPL v3.0 license and is used in compliance with its licensing terms.

The **MFD** and **MFD 2.0** resources are publicly accessible for academic research but do not clearly specify redistribution rights. As such, these lexicons are included **solely for the purpose of reproducing the experiments described in this thesis**. All rights and credit remain with the original authors.

No claim of ownership is made over these resources. Users are encouraged to obtain the original versions directly from the official sources listed below.

### Official Sources

- MFD (original): https://moralfoundations.org  
- MFD 2.0 (OSF): https://osf.io/whjt2  
- eMFD (OSF): https://osf.io/ufdcz  

### Included Datasets

This repository includes several annotated datasets used to construct and evaluate Moral Foundations Theory (MFT) classifiers. These datasets are included **solely to support reproducibility of the experiments reported in this thesis**.

- **Annotated News Articles**  
  - `coded_news.pkl`: News articles used in prior work to construct the eMFD lexicon.  
  - `highlights_raw.csv`: Human-annotated highlights indicating moral foundations and annotator information.

- **Annotated Tweets (MFTC)**  
  - `MFTC_V4.json`: Tweet IDs and corresponding Moral Foundations annotations.  
  - Some tweet texts may be unavailable due to platform policies or deletion. Where possible, original tweet texts were obtained directly from the dataset authors for research purposes.

- **Annotated Reddit Comments (MFRC)**  
  - `final_mfrc_data.csv`: Annotated Reddit comments labeled with moral foundations.

### Usage Notes

These datasets are made publicly available by their original authors for academic research. They are included here **unchanged** and **without claim of ownership**, and all rights remain with the respective creators.

The datasets are provided to enable:
- Reproduction of preprocessing steps
- Replication of model training and evaluation
- Transparency in experimental methodology

Please contact the repository's maintainers if any of these need to be removed, in line with usage policies.

### Original Sources

- Annotated news articles (OSF): https://osf.io/5r96b , https://osf.io/52qfe  
- Moral Foundations Twitter Corpus (MFTC): https://osf.io/k5n7y  
- Moral Foundations Reddit Corpus (MFRC): https://huggingface.co/datasets/USC-MOLA-Lab/MFRC


The following is taken verbatim from the original Mformer repository:

# Measuring Moral Dimensions on Social Media with Mformer

This repository accompanies the following paper:

Tuan Dung Nguyen, Ziyu Chen, Nicholas George Carroll, Alasdair Tran, Colin Klein, and Lexing Xie. **“Measuring Moral Dimensions in Social Media with Mformer”**. *Proceedings of the International AAAI Conference on Web and Social Media* 18 (2024). 

arXiv: https://doi.org/10.48550/arXiv.2311.10219.

## Check out the demo of our Mformer models via this 🤗 [Hugging Face space](https://huggingface.co/spaces/joshnguyen/mformer)!

![](/screenshots/mformer-demo.png)

## Loading Mformer locally

The 5 Mformer models are available on Hugging Face.

| Moral foundation    | Model URL |
| -------- | ------- |
| **Authority**  | https://huggingface.co/joshnguyen/mformer-authority    |
| **Care** | https://huggingface.co/joshnguyen/mformer-care     |
| **Fairness**    | https://huggingface.co/joshnguyen/mformer-fairness    |
| **Loyalty**    | https://huggingface.co/joshnguyen/mformer-loyalty    |
| **Sanctity**    | https://huggingface.co/joshnguyen/mformer-sanctity    |

Here's how to load Mformer. Note that each model's weights are in FP32 format, which totals about 500M per model. If your computer's memory does not accommodate this, you might want to load it in FP16 or BF16 format.

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# Change the foundation name if need be 
FOUNDATION = "authority"
MODEL_NAME = f"joshnguyen/mformer-{FOUNDATION}"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)
```

To perform inference:

```python
# Perform inference
instances = [
    "Earlier Monday evening, Pahlavi addressed a private audience and urged 'civil disobedience by means of non-violence.'",
    "I am a proponent of civil disobedience and logic driven protest only; not non/ irrational violence, pillage & mayhem!"
]

# Encode the instances
inputs = tokenizer(
    instances,
    padding=True,
    truncation=True,
    return_tensors='pt'
).to(model.device)

# Forward pass
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

# Calculate class probability
probs = torch.softmax(outputs.logits, dim=1)
probs = probs[:, 1]
probs = probs.detach().cpu().numpy()

# Print results
print(f"Probability of foundation {FOUNDATION}:", "\n")
for instance, prob in zip(instances, probs):
    print(instance, "->", prob, "\n")
```

which will print out the following

```bash
Probability of foundation authority:

Earlier Monday evening, Pahlavi addressed a private audience and urged 'civil disobedience by means of non-violence.' -> 0.9462048

I am a proponent of civil disobedience and logic driven protest only; not non/ irrational violence, pillage & mayhem! -> 0.97276026
```

<!-- ###  Setting up a Python environment

We will be using Anaconda. First, create an environment.

```bash
$ conda create --name moral_axes python=3.8.5
$ conda activate moral_axes
```

Install packages and dependencies.

```bash
(moral_axes) $ pip install -r requirements_deb.txt
```

## Preparing for NLP packages

### NLTK

For NLTK, download the stopwords.

```bash
$ python -m nltk.downloader stopwords
```

### spaCy

For spaCy, install the `en_core_web_md` pipeline version `3.1.0`. We need this specific version to reproduce the moral foundations news dataset (more in `mfd`).

```bash
$ python -m spacy download en_core_web_md-3.1.0 --direct
```

### GloVe

For GloVe, download the `glove.twitter.27B.200d` embedding.

```
$ cd scripts
$ mkdir data
$ mkdir data/word2vec_embeddings
$ wget https://nlp.stanford.edu/data/glove.twitter.27B.zip -P data/word2vec_embeddings/
$ unzip data/word2vec_embeddings/glove.twitter.27B.zip -d data/word2vec_embeddings/
```

Optionally, remove unused files.

```bash
$ rm data/word2vec_embeddings/glove.twitter.27B.zip
$ rm data/word2vec_embeddings/glove.twitter.27B.25d.txt
$ rm data/word2vec_embeddings/glove.twitter.27B.50d.txt
$ rm data/word2vec_embeddings/glove.twitter.27B.100d.txt
``` -->