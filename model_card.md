# Model Card: Income Prediction Model

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

&nbsp;	- Model Type: Random Forest Classifier

&nbsp;	- Framework: Scikit-learn

&nbsp;	- Model Objective: Binary classification to predict the income of individuals

&nbsp;	- Output labels: 

&nbsp;		- <= 50k

&nbsp;		- >= 50k

## Intended Use:

This model is purely for educational purposes as part of a ML DevOps pipeline project. It is not intended for any other purposes

## Training Data

The model was trained using the "Census Income (adult) dataset" found here: https://archive.ics.uci.edu/dataset/20/census+income



Catagorical Features:

\- `workclass`

\- `education`

\- `marital-status`

\- `occupation`

\- `relationship`

\- `race`

\- `sex`

\- `native-country`



Numerical Features:

\- Age

\- Education number

\- Capital gain

\- Capital loss

\- Hours per week



Target Variable:

\- `salary` (binary: `<=50K`, `>50K`)



The data was split into:

\- 80% training

\- 20% testing



Data Preprocessing

\- Categorical features were encoded using OneHotEncoder

\- Labels were transformed using LabelBinarizer

\- The same encoder and label binarizer were reused for inference to ensure consistency



Model Performance

Performance was evaluated on the held-out test dataset.



Overall Metrics

\- Precision: 0.7338

\- Recall: 0.6365

\- F1 Score: 0.6817



Slice-Based Evaluation

Performance was evaluated across slices of categorical features (e.g., education level, workclass, sex, race).



Results Saved to:

slice\_output.txt

## Evaluation Data

The model was evaluated on a test-dataset of 20% that was not used for model-training



The split was performed using a stratified train-test split based on the target variable (salary).

The evaluation data underwent the same pre-processing steps as the training data.



This evaluation data was used for model performance assessment and slice-based performance analysis across categorical features



## Metrics

The following classification metrics were used to evaluate model performance:



\- Precision: Measures the proportion of positive income predictions (`>50K`) that were correct.

\- Recall: Measures the proportion of actual high-income individuals that were correctly identified.

\- F1 Score: The harmonic mean of precision and recall, providing a balanced evaluation metric.



Overall Performance on Evaluation Data

\- Precision: 0.7338

\- Recall: 0.6365

\- F1 Score: 0.6817



Slice-Based Metrics:

In addition to overall performance, the model was evaluated across slices of categorical features (e.g., `education`, `workclass`, `sex`, `race`).



\- Slice-based precision, recall, and F1 scores were computed for each category value.

\- Results were written to `slice\_output.txt` to help identify performance variation across demographic groups.

## Ethical Considerations

\- The dataset contains demographic attributes that may be considered sensitive

\- The model may reflect historical biases present in the data

\- Predictions should not be used to make decisions that impact individuals’ livelihoods or access to opportunities

\- Slice-based evaluation was conducted to help identify performance inconsistencies across groups, but this does not eliminate bia.



## Caveats and Recommendations

\- Avoid real-world deployment without further bias mitigation and fairness analysis.

\- Consider additional evaluation metrics and fairness constraints.

\- Retrain the model with updated or more representative data if used beyond demonstration.

