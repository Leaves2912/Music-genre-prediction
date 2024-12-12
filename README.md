# Music Genre Prediction
This project uses machine learning to classify music genres from audio files. It trains a Random Forest model on the GTZAN dataset and provides real-time predictions through a Streamlit interface, combining efficient preprocessing, feature extraction, and user-friendly interaction.

## INSTALLATION INSTRUCTIONS
Follow the steps below to set up and run the Audio Genre Prediction project :

### Step 1: Clone the Repository :
Clone the repository containing the project files using the following command :

          git clone <repository_url>
          cd <repository_name>
          
### Step 2 : Create a Virtual Environment (Optional but Recommended)
Create and activate a virtual environment to manage dependencies :
          
          python -m venv env
On windows -
          
          env\Scripts\activate

On macOS/Linux -

          source env/bin/activate

### Step 3 : Install the required packages :
In this project, GTZAN dataset has been used. The dataset should be available in the system before running the code.

Install all the libraries and modules listed in requirements.txt by using the following command -

          pip install -r requirements.txt

### Step 4 : Prepare the Dataset
Place your dataset file (`features_30_sec.csv`) in the specified path. Update the path in the TRAIN MODULE code if necessary.

### Step 5 : Train the Model
Run the training script to train the Random Forest model and save it :
          
          python train_module.py
The trained model will be saved as 'best_rf_model.pkl' in the project directory.

### Step 6 : Run the Streamlit Application
Start the Streamlit application to predict audio genres :

          streamlit run test_module.py

### Step 7 : Test with an Audio File
Use the application interface to enter the URL of an audio file. The system will download the audio, extract its features, and predict its genre.

## Usage Instructions

1. TRAIN MODULE: Run the train_module.py script to train a Random Forest model on the provided audio features dataset (features_30_sec.csv). The trained model is saved as best_rf_model.pkl.
2. TEST MODULE: Run the test_module.py script to launch a Streamlit-based application for predicting audio genres.
3. Enter the URL of an audio file in the Streamlit app to download and process the file.
4. The app extracts audio features, uses the pre-trained model, and displays the predicted genre.
5. Ensure the dataset and best_rf_model.pkl are correctly placed and accessible before running the scripts.
