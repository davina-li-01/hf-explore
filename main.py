import time
from transformers import pipeline

start = time.time()
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print(f"Model loaded in {time.time() - start:.1f}s\n")

labels = ["technical skill", "soft skill", "experience requirement", "education requirement"]

job_listing_lines = [
    "Proficiency in SQL and Python for building dashboards and automating reports.",
    "Strong communicator who can present findings to senior stakeholders.",
    "2+ years of experience in management consulting or business analytics.",
    "Bachelor's degree in Business, Economics, or a related field.",
]

for line in job_listing_lines:
    t = time.time()
    result = classifier(line, candidate_labels=labels)
    top_label, top_score = result["labels"][0], result["scores"][0]
    print(f"{line}\n  -> {top_label} ({top_score:.0%})  [{time.time() - t:.2f}s]\n")