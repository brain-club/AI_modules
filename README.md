# AI_modules

## Requirements
For `offensiveness_censoring.py`,

Install `profanity-check` library by using the command below.
```
pip install profanity-check
```

For `Organizer.py`,

Install `sentence-transformers` library, `Pytorch`, and `numpy` by using the command below.
```
pip install sentence-transformers torch numpy
```


## Offensiveness Censoring
Based on library from https://github.com/vzhou842/profanity-check

Python file `offensiveness_censoring.py` containing function `censoring()`


  - **INPUT**: (list) sentences to be checked
  
  - **OUTPUT**: (list) booleans(true/false) if the sentence has offensiveness or profanity
  
  
## Organizer
Using library from https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens?text=The+goal+of+life+is+%5BMASK%5D.

Python file `organizer.py` containing function `organizer()`

  - **INPUT**: (dict) sentence ids as keys and sentences as its values
  
  - **OUTPUT**: (int) number of clusters, (dict) sentence ids as keys and assigned cluster number as its value
  
