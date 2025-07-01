import pandas as pd

# Load CSV files
df_train = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\Training.csv")
df_description = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\description.csv")
df_diets = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\diets.csv")
df_medications = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\medications.csv")
df_precautions = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\precautions_df.csv")

# Print column names to check
from huggingface_hub import model_info

model_name = "manoj2423/medhyd"
info = model_info(model_name)
print(info.siblings)  # Lists all files in the repository

from huggingface_hub import login

app = Flask(__name__)

login(token="") # Add your hugging face login token here

model_name = "manoj2423/medi"
subfolder = "model"  # Explicitly load from this subfolder

tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder=subfolder)
model = AutoModelForCausalLM.from_pretrained(model_name, subfolder=subfolder)



df_train = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\Training.csv")
df_description = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\description.csv")
df_diets = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\diets.csv")
df_medications = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\medications.csv")
df_precautions = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\precautions_df.csv")
df_workout = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\workout_df.csv")




df_train = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\Training.csv", sep=",")
df_description = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\description.csv", sep=",")
df_diets = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\diets.csv", sep=",")
df_medications = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\medications.csv", sep=",")
df_precautions = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\precautions_df.csv", sep=",")
df_workout = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\workout_df.csv", sep=",")
df_symptoms = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\symptoms_df.csv", sep=",")
df_severity = pd.read_csv(r"C:\Users\smano\OneDrive\Desktop\hyd3\datasets\Symptom-severity.csv", sep=",")
