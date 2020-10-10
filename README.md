# AI_modules

## Requirements
For `offensiveness_censoring.py`,

Install `profanity-check` library by using the command below.
```
pip install profanity-check
```

For `Organizer.py`,

Install `pytorch-transformers` library by using the command below.
```
pip install pytorch-transformers
```


## Offensiveness Censoring
Based on library from https://github.com/vzhou842/profanity-check

Python file `offensiveness_censoring.py` containing function `censoring()`


  - **INPUT**: list of sentences to be checked
  
  - **OUTPUT**: list of booleans(true/false) if the sentence has offensiveness or profanity
  
  
## Organizer
