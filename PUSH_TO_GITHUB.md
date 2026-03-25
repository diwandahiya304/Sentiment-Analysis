# 🚀 How to Push This Project to GitHub

Follow these steps to upload all files to your existing repo:
**https://github.com/diwandahiya304/Sentiment-Analysis**

---

## Step 1 — Clone Your Existing Repo

```bash
git clone https://github.com/diwandahiya304/Sentiment-Analysis.git
cd Sentiment-Analysis
```

---

## Step 2 — Copy All New Files Into the Cloned Folder

Copy everything from the downloaded zip into the cloned folder.
Your folder should now look like this:

```
Sentiment-Analysis/
├── notebooks/
│   └── hotel_sentiment_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

> ⚠️ You can **delete** the old duplicate notebooks:
> - `sentiment analysis.ipynb`
> - `sentiment analysisfinal.ipynb`
> - `sentiment_analysis-Copy1.ipynb`
> - `sentiment_analysis.ipynb`
>
> Keep only `notebooks/hotel_sentiment_analysis.ipynb`

---

## Step 3 — Stage All Changes

```bash
# Remove old duplicate notebooks
git rm "sentiment analysis.ipynb"
git rm "sentiment analysisfinal.ipynb"
git rm "sentiment_analysis-Copy1.ipynb"
git rm "sentiment_analysis.ipynb"

# Add all new files
git add .
```

---

## Step 4 — Commit

```bash
git commit -m "feat: restructure project — add src modules, clean notebook, README, requirements"
```

---

## Step 5 — Push

```bash
git push origin main
```

If your default branch is `master`, use:
```bash
git push origin master
```

---

## Step 6 — Add the Dataset (Optional)

The CSV is excluded from Git (see `.gitignore`) because it's large.
You can share it via:
- **Google Drive link** — already in your README
- **Git LFS** — for large file tracking:
  ```bash
  git lfs install
  git lfs track "data/*.csv"
  git add .gitattributes
  git add data/hotel_reviews.csv
  git commit -m "add: dataset via Git LFS"
  git push
  ```

---

## ✅ Done!

Your repo will have a clean, professional structure ready to share.
