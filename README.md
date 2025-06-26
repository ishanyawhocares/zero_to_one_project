I have build a Review Sentiment Analyzer for a Movie. It takes the review of a movie and gives the review if the review is Positive, Negative or Neutral.

Main Page:
![image](https://github.com/user-attachments/assets/f06d6020-f229-435c-b8f0-5ef33a84cc47)

The result page:
![image](https://github.com/user-attachments/assets/62626cc1-b7ad-4128-be04-3422427426d5)

The link to the website : 
https://ishanyawhocares.github.io/zero_to_one_project/templates

Link to the ML code :
https://colab.research.google.com/drive/1XQjZJbKXTM3HWyYzSinxLf5DwYt4048y?usp=sharing

How to run the code :
Import the file on your personal computer, then run Index.html.

Brekdown of how I made this :

1. Dataset Overview

I used a combined dataset of 52,000 movie reviews. Here's the breakdown:

    50,000 came from the well-known IMDb dataset—these were already labeled as "positive" or "negative."

    2,000 were custom-written reviews I manually labeled as "neutral." I added these to help the model handle more subtle sentiments that aren't clearly good or bad.

Mixing and shuffling the data helped keep the distribution natural and avoided overfitting on any one source.
2. Preprocessing the Text

Before throwing the reviews into the model, I ran them through a cleaning pipeline—pretty standard stuff, but crucial.

Used NLTK for this. The steps:

    Lowercasing everything to avoid treating "Great" and "great" as different.

    HTML cleaning to strip tags like <br />, which IMDb reviews often contain.

    Removed punctuation and weird characters—basically anything that's not a letter.

    Stopword filtering using NLTK’s default list (e.g., "the", "is", "in")—they don’t add much value.

    Lemmatization, not stemming—this is important! It turns "running" into "run" but keeps "better" as "good", so it retains meaning better.

I may revisit the lemmatization rules later, just to see if anything slips through.
3. Turning Text Into Numbers

Once the text was clean, I needed to feed it into a model, so I used TensorFlow’s Tokenizer.

    Tokenization: Built a vocab of the 10,000 most frequent words (arbitrary cutoff but works well in practice). Each word becomes an integer.

    Padding: Made sure all sequences are 100 tokens long—short ones get zero-padded, long ones get chopped off.

I didn’t experiment with variable-length input yet, but that might be interesting to try if performance plateaus.
4. Model Architecture

Here’s the model I built with Keras (Sequential style):
Layer	Output Shape	Notes
Input	(None, 100)	Takes in the padded sequences
Embedding	(None, 100, 128)	Maps word indices to dense vectors (128D)
Bidirectional LSTM	(None, 128)	Reads forward and backward—helps with context
Dense (Hidden)	(None, 64)	Adds some abstraction to the features
Dropout (0.5)	(None, 64)	Regularization—randomly drops half the neurons
Output (Dense)	(None, 3)	Final classifier (softmax for 3 classes)

The bidirectional LSTM is the real MVP here—it catches patterns from both ends of the sentence, which really helps in text-heavy tasks.
5. Training & Evaluation

Trained on 80% of the data, validated on the remaining 20%.

    Optimizer: Adam (default settings—might tweak later)

    Loss function: categorical_crossentropy, since it’s a 3-class problem.

    EarlyStopping: Used to stop training once the validation accuracy flatlines. Helps avoid overfitting, and saves the best checkpoint automatically.

So far, accuracy looks solid. Next steps would probably involve testing on completely out-of-sample data and maybe looking at confusion matrices to see where it gets confused.
