# ai-fraud-detection

## Output of script.py

When you run `script.py`, you will see the following outputs:

1. **Classification Report**: After training the logistic regression model, the script prints a classification report showing precision, recall, f1-score, and support for each class (fraud/legit).

2. **Confusion Matrix**: The script prints the confusion matrix for the test set predictions.

3. **Prediction Example**: The script runs a sample transaction through the model and prints whether it is predicted as FRAUD or LEGIT, along with the probability.

**Example output:**
```
=== Classification Report ===
  precision    recall  f1-score   support

           0       0.98      0.99      0.99      2000
           1       0.92      0.85      0.88       200

    accuracy                           0.98      2200
   macro avg       0.95      0.92      0.93      2200
weighted avg       0.98      0.98      0.98      2200

=== Confusion Matrix ===
[[1980   20]
 [  30  170]]
Prediction: LEGIT (probability: 0.12)
```
*Note: The actual numbers will vary depending on your data and model.*