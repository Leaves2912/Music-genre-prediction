# Music-genre-prediction
This project uses machine learning to classify music genres from audio files. It trains a Random Forest model on the GTZAN dataset and provides real-time predictions through a Streamlit interface, combining efficient preprocessing, feature extraction, and user-friendly interaction.

## INSTALLATION INSTRUCTIONS -
Follow the steps below to set up and run the Audio Genre Prediction project :

### Step 1: Clone the Repository :
Clone the repository containing the project files using the following command :

          git clone <repository_url>
          cd <repository_name>
          
### Step 2 : Create a Virtual Environment (Optional but Recommended)
Create and activate a virtual environment to manage dependencies :
          
          python -m venv env
(On Windows - `env\Scripts\activate`)

(On macOS/Linux - `source env/bin/activate`)

### Step 3 : Install the required packages : 
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
