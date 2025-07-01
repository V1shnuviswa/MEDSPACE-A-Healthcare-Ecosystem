from flask import Flask, request, render_template, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

app = Flask(__name__)

# Login and load model
login(token="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")
model_name = "manoj2423/medi"
subfolder = "model/checkpoint-14649"

tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder=subfolder)
model = AutoModelForCausalLM.from_pretrained(model_name, subfolder=subfolder)

@app.route("/symptom", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            symptoms = request.form.get("symptoms", "").strip()
            if not symptoms:
                return render_template("resultsy.html", info="No symptoms provided!")

            # Prepare input text
            input_text = f"Symptoms: {symptoms}\nPredict the disease and recommend treatment."
            inputs = tokenizer(input_text, return_tensors="pt")

            # Generate prediction
            with torch.no_grad():
                output = model.generate(**inputs, max_length=150)

            response_text = tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract structured data
            medications = response_text.split("Medications:")[-1].split("\n")[0] if "Medications:" in response_text else "Not provided"
            diet = response_text.split("Diet:")[-1].split("\n")[0] if "Diet:" in response_text else "Not provided"
            precautions = response_text.split("Precautions:")[-1].split("\n")[0] if "Precautions:" in response_text else "Not provided"

            return render_template(
                "resultsy.html",
                symptoms=symptoms,
                disease_prediction=response_text.split("\n")[0],  # Extract first line as disease info
                medications=medications,
                diet=diet,
                precautions=precautions,
            )

        except Exception as e:
            return render_template("resultsy.html", info=f"Prediction failed: {e}")

    return render_template("indexsy.html")

if __name__ == "__main__":
    app.run(debug=True)
