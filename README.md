# Toxic Comment Classification Challenge
**Link:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge#description

**Description:**
Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.
The [Conversation AI](https://conversationai.github.io/) team, a research initiative founded by [Jigsaw](https://jigsaw.google.com/) and Google (both a part of Alphabet) are working on tools to help improve online conversation. One area of focus is the study of negative online behaviors, like toxic comments (i.e. comments that are rude, disrespectful or otherwise likely to make someone leave a discussion). So far they’ve built a range of publicly available models served through the [Perspective API](https://perspectiveapi.com/), including toxicity. But the current models still make errors, and they don’t allow users to select which types of toxicity they’re interested in finding (e.g. some platforms may be fine with profanity, but not with other types of toxic content).
In this competition, you’re challenged to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s [current models](https://github.com/conversationai/unintended-ml-bias-analysis). You’ll be using a dataset of comments from Wikipedia’s talk page edits. Improvements to the current model will hopefully help online discussion become more productive and respectful.

**Evaluation:**
Submissions are evaluated on the mean column-wise [log loss](https://www.kaggle.com/wiki/LogLoss). In other words, the score is the average of the log loss of each predicted column.
**Submission File:**
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

    id,toxic,severe_toxic,obscene,threat,insult,identity_hate
    6044863,0.5,0.5,0.5,0.5,0.5,0.5
    6102620,0.5,0.5,0.5,0.5,0.5,0.5
    etc.


**Group name candidates:**  

- **Phantom Brigade** 

**Group Rule:**

1. **Team members share codes here**
2. **Every person creates his/her own folder and put codes there.**
3. **Do not uploading training or testing datasets.**
4. **All the datasets must be moved to a folder with name "data"**
5. **All the executable files must be moved to a folder with name "apps”**

**[Discussion paper](https://paper.dropbox.com/doc/New-Ideas-Post-here-bIBrgHTcmLzXW3iNS7OiS)**

**[Word2Vec pretrained models](http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings/)**

**[Google Colab free GPU tutorial](https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d)**


