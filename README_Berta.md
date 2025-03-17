Model Training and Evaluation
To detect fraudulent transactions effectively, we employed two machine learning models: Logistic Regression and XGBoost. Given the severe class imbalance, special techniques were used to enhance model performance.

Logistic Regression
Logistic Regression is a simple yet powerful model for binary classification. To address the class imbalance, the class_weight="balanced" parameter was used, ensuring the minority class (fraudulent transactions) received appropriate weight. The model was trained with 1000 iterations for optimal convergence.

XGBoost Classifier
XGBoost is a high-performance gradient boosting algorithm known for its efficiency in handling imbalanced datasets. The scale_pos_weight parameter was used to balance fraud detection, and the binary:logistic objective was chosen for probability-based predictions.

Evaluation Metrics
Since fraud detection requires prioritizing correctly identifying fraudulent cases, we used:

Recall: Measures the percentage of fraudulent transactions correctly identified. Higher recall reduces false negatives.
Precision-Recall AUC (PR AUC): Evaluates model performance on imbalanced datasets, focusing on fraud detection rather than overall accuracy.
Results
Logistic Regression provided the best performance, achieving a high recall score while maintaining strong precision-recall balance. This indicates its effectiveness in detecting fraudulent transactions without excessive false positives.
