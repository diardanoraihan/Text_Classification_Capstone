# Deep Learning Techniques for Text Classification
Master Thesis Project at Nanyang Technological University
- __Author__: Diardano Raihan 
- __Email__: diardano@gmail.com
- __Social Media__: [LinkedIn](https://www.linkedin.com/in/diardanoraihan), [Medium](https://diardano.medium.com/)

## Project Summary
The experiment will evaluate the performance of some popular deep learning models, such as feedforward, recurrent, convolutional, and ensemble-based neural networks, on five different datasets. We will build each model on top of two separate feature extractions to capture information within the text. 

The result shows that the __word embedding provides a robust feature extractor__ to all the models in making a better final prediction. The experiment also highlights __the effectiveness of the ensemble-based__ and __temporal convolutional neural network__ in achieving good performances and even competing with the state-of-the-art benchmark models.

## The Proposed Deep Learning Models
| Model | Bag-of-Words | WE-avg| WE-random | WE-static | WE-dynamic |
| --- | --- | --- | --- | --- | --- |


## Datasets
| Dataset | Classes | Average <br /> Sentence Length | Dataset Size | Vocab Size | Test Size | 
|:-------:|:-------:|:-------------------:|:------------:|:----------:|:---------:|
| `MR`    | 2       | 20                  | 10662        | 18758      | CV        |
| `SUBJ`  | 2       | 23                  | 10000        | 21322      | CV        |
| `TREC`  | 5       | 10                  | 5952         | 8759       | 500       |
| `CR`    | 2       | 19                  | 3775         | 5334       | CV        |
| `MPQA`  | 2       | 3                   | 10606        | 6234       | CV        |
- __MR__. Movie Reviews – classifying a review as positive or negative [17].
- __SUBJ__. Subjectivity – classifying a sentence as subjective or objective [18].
- __TREC__. Text REtrieval Conference – classifying a question into six categories (a person, location, numeric information, etc.) [19].
- __CR__. Customer Reviews – classifying a product review (cameras, MP3s, etc.) as positive or negative [20].
- __MPQA__. Multi-Perspective Question Answering – opinion polarity detection [21].
_Note_: the test size CV stands for cross-validation. It indicates the original dataset does not have a standard train/test split. Hence, we use a 10-fold CV. The AcademiaSinicaNLPLab [22] repository provides access to all these datasets.

## Training Process



# Result and Discussion







