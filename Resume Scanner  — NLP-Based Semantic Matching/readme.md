# Resume Scanner  — NLP-Based Semantic Matching

This smarter version of the resume scanner uses natural language processing (NLP) to extract resume content from PDFs and match it against a job description with expanded keyword recognition—including synonyms and related terms.

Instead of strict keyword comparison, it evaluates semantic relevance using spaCy and WordNet.

---

## 📌 Project Overview

Recruiters look for resumes that match job requirements—but candidates may use different wording to describe the same skill. This scanner expands job keywords using synonyms from WordNet, processes resumes with spaCy, and calculates a match score that reflects deeper relevance.

---

## 🧰 Key Features

-  Reads resume text from PDFs (`pdfminer.six`)
-  Expands job keywords using WordNet synonyms
-  Tokenizes resume text with `spaCy` NLP pipeline
-  Compares resumes and assigns match scores
-  Highlights matched & missing concepts clearly

---

## Results
![manu 3](https://github.com/user-attachments/assets/e54592e2-8e7f-4343-85e8-3afae4dd8891)
