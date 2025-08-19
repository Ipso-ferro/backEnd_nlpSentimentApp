1. Boundaries
1.1. Input scope
The data source is from the folder "Data" with different folders (books, dvd, electronics, kitchen&houseware) that are categorized into positive and negative reviews

1.2. Text format
- All reviews are standardized by blank lines
If the chunks are fewer than 10, it will fall back to the line by line splitting. More details (def _read_reviews from train.py file)
- All text are in lowercase

1.3. Output
The output is configured to display the sentiment result in the web UI: either positive or negative with a confidence of percentage

1.4. API endpoints
- POST request: accepts JSON file, extract and clean the text, then validate the input, and return the results with format "sentiment:..., confidence:..."

2. Fairness
2.1. Data bias
- The model is using Logistic Regression with balanced class weight to handle the imbalanced of the negative and positive reviews. This ensures that the model is trained with equal weighting of class
- Use TfidfVectorizer to process the reviews. However, considering the cultural and linguisitic aspects, the model might not be able to process human jokes, slangs, or sarcasm. 
- Also all reviews are in English, therefore, other languages are not supported

2.2. Neutral sentiment
The binary classification (only positive or negative) cannot process a neutral reviews, mixed sentiments, and might give wrong results.

3. Limitations
- The model only predicts binary results (positive or negative sentiment), not able to predict neutral or mixed reviews
- Data source: relies on the folder Data, if the folder is missing, the model cannot be executed and the training fails. Also, if the data is not balanced (one category dominated the whole dataset), then the model will overfit its vocabulary, resulting less fairness.
- Scalability: the endpoint can only handle one review at a time
- The code should have loggings of prediction performance or bias metrics to detect and debug errors
- Algorithm choices: might consider better algorithms such as DistilBERT which could capture word relationship 
- No stopword removal is included to remove unnecessary words
