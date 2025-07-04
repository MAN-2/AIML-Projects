# 🎬 Movie Rating Predictor — User-to-User Collaborative Filtering 

This beginner-friendly project predicts how a user might rate a selected movie based on historical user ratings using cosine similarity. It also recommends movies similar to the one entered, all with actual movie names sourced from the MovieLens dataset.

---

## 📌 Features

- 📄 Predict user ratings for any movie in the dataset
- 👤 Uses collaborative filtering via user-user similarity
- 🎞️ Displays top 5 similar movies based on viewer overlap
- ✅ Uses only pandas and sklearn (no external ML libraries)
- 💡 Understandable logic and easy to customize

---

## 🧠 How It Works

1. Loads and merges `ratings.csv` and `movies.csv` from the MovieLens dataset.
2. Builds a **User-Item Matrix** of ratings.
3. Calculates cosine similarity between users to measure preference alignment.
4. Predicts rating for the target user and movie using a weighted average of ratings by similar users.
5. Computes similarity between movies and recommends top matches based on user rating overlap.

---
 
## Results:
![manu 4](https://github.com/user-attachments/assets/75187841-5f29-481b-b563-efe260be1dfd)




## 📦 Installation

```bash
pip install pandas scikit-learn
