import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Model cache
llama_model = None
llama_tokenizer = None

# Optional: set a custom cache directory if needed
custom_cache_dir = "/cluster/scratch/gcardenal/LLM_models"

def load_llama_8b_model():
    """
    Loads the local Llama-8B model if not already loaded.
    """
    global llama_model, llama_tokenizer

    if llama_model is None or llama_tokenizer is None:
        print("Loading Llama-8B model locally...")
        llama_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct", 
            cache_dir=custom_cache_dir
        )
        llama_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            cache_dir=custom_cache_dir,
            device_map="auto",
            torch_dtype="auto"
        )

def run_llama_8b_inference(question, system_prompt):
    """
    Llama 8B is local only. No API calls are attempted.
    Returns the generated answer as a string.
    """
    load_llama_8b_model()

    print("Starting inference for question (Llama-8B):", question[:50])

    pipe = pipeline(
        "text-generation",
        model=llama_model,
        tokenizer=llama_tokenizer,
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    messages_templated = llama_tokenizer.apply_chat_template(messages, tokenize=False)

    raw_output = pipe(messages_templated, max_new_tokens=1024)
    generated_text = raw_output[0]["generated_text"]

    pattern = r'^<\|begin_of_text\|>.*?<\|eot_id\|>assistant\s*'
    cleaned_answer = re.sub(pattern, '', generated_text, flags=re.DOTALL).strip()

    return cleaned_answer

# Load data
with open("patient_summaries_top28.json") as f:
    data = json.load(f)

# Few-shot prompt
few_shot_prompt = """
Task: Predict whether the patient will be discharged alive on a scale from 1 (certain death) to 10 (certain survival).

Examples:

Patient: 35-year-old patient weighing 70.0 kg on mechanical ventilation, GCS 15.0 (stable), variability 0.00. HR 80 bpm (variability: normal). Respiratory rate variability: stable. Glucose 90.0 mg/dL, BUN 14.0 mg/dL, Lactate 1.0 mmol/L (clearance: 10.0%). HCO₃ 24.0 mmol/L, pH 7.40, PaCO₂ 40.0 mmHg. Na 140.0 mEq/L, Mg 2.0 mmol/L. WBC 7.0 cells/nL, Platelets 250 cells/nL, ALT 30.0 IU/L, SaO₂ 98% (stable), FiO₂ 0.21, PF ratio min 450.0, urine output 1500 mL, NIMAP 85.0 mmHg, NIDiasABP 70.0 mmHg.
Prediction: 10

Patient: 100-year-old patient weighing 0.7 kg on mechanical ventilation, GCS 3.0 (unresponsive), variability 0.01. HR 0 bpm (variability: none). Respiratory rate variability: absent. Glucose 5.0 mg/dL, BUN 65.0 mg/dL, Lactate 9.0 mmol/L (clearance: 0.0%). HCO₃ 3.0 mmol/L, pH 6.75, PaCO₂ 18.0 mmHg. Na 120.0 mEq/L, Mg 0.4 mmol/L. WBC 1.0 cells/nL, Platelets 4 cells/nL, ALT 1300.0 IU/L, SaO₂ 60% (decreasing), FiO₂ 1.0, PF ratio min 50.0, urine output 10 mL, NIMAP 30.0 mmHg, NIDiasABP 12.0 mmHg.
Prediction: 1

Output Format  
You must provide the evaluation **exactly** in the following format.  
**Do not generate anything else besides the score.**  
The score should be a single integer between 1 (certain death) and 10 (certain survival).  

Prediction:$Score
"""

# Run predictions
scores = []

for item in data:
    prompt = f"\n\nNow evaluate the following patient:\n\nPatient: {item['summary_text']}\nPrediction:$Score"
    response = run_llama_8b_inference(prompt, few_shot_prompt)

    try:
        content = response.strip()
        print("Model response:", content)

        # Extract the value after 'Prediction:'
        for line in content.splitlines():
            if line.strip().startswith("Prediction:"):
                pred = int(line.strip().split(":")[1])
                break
        else:
            raise ValueError("No prediction line found.")

        pred = max(1, min(pred, 10))  # clip to [1, 10]

    except Exception as e:
        print("Failed to parse prediction, defaulting to 5. Error:", e)
        pred = 5  # fallback to neutral prediction

    scores.append(pred / 10)  # normalize to [0,1]
    print("Normalized score:", pred / 10)

# Save scores to a JSON file
with open("llama_normalized_scores.json", "w") as f:
    json.dump(scores, f, indent=2)

# Or optionally save as CSV
pd.DataFrame(scores, columns=["normalized_score"]).to_csv("llama_normalized_scores.csv", index=False)