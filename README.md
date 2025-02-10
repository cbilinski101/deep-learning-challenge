![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)

# **Alphabet Soup Deep Learning Model Performance Report**

---

## **Overview of the Analysis**

The purpose of this analysis is to predict whether applicants will be successful if funded by the nonprofit foundation **Alphabet Soup**. By leveraging deep learning techniques, we aim to develop a model that can accurately classify potential success, aiding the foundation in making informed funding decisions.

---

## **Results**

### *Data Preprocessing*

- **Target Variable:**  
  - `IS_SUCCESSFUL`

- **Feature Variables:**  
  - `APPLICATION_TYPE`
  - `AFFILIATION`
  - `CLASSIFICATION`
  - `USE_CASE`
  - `ORGANIZATION`
  - `STATUS`
  - `INCOME_AMT`
  - `SPECIAL_CONSIDERATIONS`
  - `ASK_AMT`

- **Variables Removed from Input Data:**  
  - `EIN`
  - `NAME`

### *Compiling, Training, and Evaluating the Model*

- **Model Architecture:**
  - **Hidden Layer 1:** 128 neurons, activation function: *tanh*
  - **Hidden Layer 2:** 64 neurons, activation function: *relu*
  - **Hidden Layer 3:** 32 neurons, activation function: *relu*
  - **Hidden Layer 4:** 16 neurons, activation function: *relu*
  - **Hidden Layer 5:** 8 neurons, activation function: *relu*
  - **Hidden Layer 6:** 4 neurons, activation function: *relu*
  - **Hidden Layer 7:** 2 neurons, activation function: *sigmoid*

  I used a total of 7 layers with the number of neurons starting at 128 and decreasing to 2. The combination of activation functions (*tanh*, *relu*, *sigmoid*) was selected to handle different aspects of data complexity and non-linearity. Using **PCA** to visualize the data revealed significant overlap, suggesting a need for a more complex architecture to achieve the 75% target accuracy.

  **PCA Scatter Plot:**  
![image](https://github.com/user-attachments/assets/33505f8a-7589-4191-9b73-bc9f987fff30)

- **Model Performance:**
  - **Accuracy Achieved:** *75.25%*  
  Considering the overlapping nature of the data, this performance is impressive and meets the target threshold.

- **Steps Taken to Improve Model Performance:**
  - Created a **correlation matrix** to identify and drop highly correlated columns that could skew results.
  
  **Correlation Matrix:**  
![image](https://github.com/user-attachments/assets/47537f36-6672-4de0-8bbc-87384c0a58d2)

  - Focused on numerical columns, particularly `ASK_AMT`, and applied **IQR** to detect and remove outliers.
  - Filtered data based on **IQR** results to ensure a cleaner dataset for model training.

---

## **Summary**

The final model achieved an accuracy of **75.25%**, surpassing the target accuracy threshold. The model’s performance metrics are as follows:

- **Loss:** `0.5313`
- **Accuracy:** `0.7525`

Given the overlapping nature of the dataset, achieving over 75% accuracy indicates that the chosen architecture and preprocessing steps were effective. However, there is room for improvement. 

### **Recommendation:**  
For future iterations, I recommend experimenting with ensemble methods like **Random Forest** or **Gradient Boosting**. These models can handle classification problems effectively, especially when dealing with overlapping classes. Additionally, **hyperparameter tuning** and **cross-validation** could further optimize model performance. Incorporating more advanced **feature engineering** techniques or utilizing **SMOTE** for balancing the dataset might also improve classification accuracy.

---

### **Project Setup**

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/AlphabetSoupCharity.git
    cd AlphabetSoupCharity
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install required dependencies:**
    ```bash
    pip install numpy pandas matplotlib seaborn tensorflow
    ```

4. **Run the notebooks:**
    Open the notebooks using Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab AlphabetSoupCharity.ipynb
    jupyter lab AlphabetSoupCharity_Optimization.ipynb
    ```

### **Project Structure**

```plaintext
├── AlphabetSoupCharity.ipynb
├── AlphabetSoupCharity_Optimization.ipynb
├── README.md
├── data/
│   └── charity_data.csv
└── screenshots/
    ├── screenshot1.png
    └── screenshot2.png
```

### **License**
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

This project was developed with the assistance of the following resources:

- **"U of T" (Turtoring Session - Xpert Learning Assistant - GitLab Activites)** – Provided guidance on code and explanations.
- **ChatGPT** – Assisted with code, explanations, and README formatting. 
