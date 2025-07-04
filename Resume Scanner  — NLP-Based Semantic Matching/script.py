import os
import re
from pdfminer.high_level import extract_text
import spacy
from nltk.corpus import wordnet
import nltk


nltk.download('wordnet') # for synonym expansion


nlp = spacy.load("en_core_web_sm") # model 

# Expand keywords using WordNet synonyms
def expand_keywords(text):
    base_keywords = set()
    synonyms = set()
    
    for token in nlp(text):
        word = token.text.lower()
        if token.is_alpha:
            base_keywords.add(word)
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().lower())
                    
    return base_keywords.union(synonyms)

#  extract lowercase text
def read_pdf(file_path):
    try:
        return extract_text(file_path).lower()
    except:
        return ""

#  job description
job_desc = "Looking for a data scientist skilled in Python, machine learning, SQL and data analysis."
job_keywords = expand_keywords(job_desc)

# local path for Resume
resume_folder = r"E:/project/AIML project showcase/Resume Scanner using Keyword/sample resumes"
results = []

for file in os.listdir(resume_folder):
    if file.endswith(".pdf"):
        path = os.path.join(resume_folder, file)
        resume_text = read_pdf(path)
        resume_tokens = set(token.text.lower() for token in nlp(resume_text) if token.is_alpha)
        matched = job_keywords.intersection(resume_tokens)
        missing = job_keywords - resume_tokens
        score = len(matched)

        if score >= 10:
            level = " Strong  match"
        elif score >= 5:
            level = " Moderate match"
        else:
            level = " Weak match"

        results.append({
            "name": file,
            "score": score,
            "strength": level,
            "matched": matched,
            "missing": missing
        })

# Print enhanced output
print(" Semantic Resume Match Results:\n")
for r in sorted(results, key=lambda x: x["score"], reverse=True):
    print(f" Resume: {r['name']}")
    print(f" Score: {r['score']}")
    print(f" Match Level: {r['strength']}")
    print(f" Matched Terms: {', '.join(sorted(r['matched'])) if r['matched'] else 'None'}")
    print(f" Missing Terms: {', '.join(sorted(r['missing'])) if r['missing'] else 'None'}")
    print("-" * 50)
