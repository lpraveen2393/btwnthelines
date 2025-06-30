# btwn the lines 🧠

**btwn the lines** is a machine learning-based application that analyzes relationship chat conversations to detect signs of emotional manipulation, toxicity, or healthy communication patterns. It identifies potential culprits and victims, highlights behavioral patterns, and provides personalized advice for each partner.

---

## ✨ Key Features

- Classifies relationship dynamics as **Healthy**, **Manipulative**, **Toxic**, or **Both**
- Identifies the likely **culprit** and **victim** from the conversation
- Detects common **manipulation techniques** and **emotional vulnerabilities**
- Provides tailored, behavior-aware advice for both partners
- Combines machine learning with rule-based emotional cues
- Speaker-wise behavioral analysis

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/btwn-the-lines.git
cd btwn-the-lines
```

### 2. Install Dependencies
Ensure Python 3.9 or later is installed on your system.
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run streamlit_app.py
```

This will launch the app in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
btwn-the-lines/
├── streamlit_app.py              # Main Streamlit app file
├── relationship_chat_model.pkl   # Pretrained ML model (auto-loaded if available)
├── mentalmanip_con.csv           # (Optional) Training dataset
├── requirements.txt              # List of required Python packages
└── README.md
```

---

## 🧠 Model Details

The model includes:
- TF-IDF features + engineered features (e.g., indicator ratios, punctuation use)
- Random Forest Classifier trained on labeled chat samples
- SMOTE for class balancing
- Rule-based logic for improved classification confidence and emotional insights

To retrain the model (if `relationship_chat_model.pkl` is missing), run:
```bash
python streamlit_app.py --train
```

---

## 💬 Chat Input Format

You can upload or paste chats in the following format:
```
PersonA: I'm really sorry.
PersonB: You always say that, but nothing changes.
PersonA: Please don’t be mad. I’ll do better.
```

---

## 📸 Screenshots

Screenshots of the app and its output are available here:  
**🔗 [View Screenshots](https://drive.google.com/drive/folders/1uFx2f_GBPPGF8LJRu5zAxG_uC9WhWbLT?usp=sharing)**  
_(Replace this with the actual URL or GitHub folder path where you host the images)_

---

## 📃 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for full terms.

---


