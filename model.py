import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Keyword dictionaries (unchanged)
HEALTHY_INDICATORS = [
    "understand", "listen", "respect", "appreciate", "sorry", "thank", 
    "feel", "together", "support", "communicate", "agree", "help",
    "love", "care", "important to me", "i understand", "let's talk"
]

MANIPULATIVE_INDICATORS = [
    "if you loved me", "if you cared", "after all i've done", 
    "everyone thinks", "nobody would", "you're overreacting",
    "you're too sensitive", "not that bad", "i'll leave you",
    "can't live without you", "you owe me", "normal people",
    "you made me", "for your own good", "why can't you just",
    "you're imagining things", "remember when you", "if you really",
    "i'm the only one", "no one else would", "look what you made me do"
]

TOXIC_INDICATORS = [
    "you always", "you never", "shut up", "stupid", "kill yourself", "idiot", 
    "hate you", "worthless", "useless", "pathetic", "loser", "dumb", 
    "ugly", "fat", "crazy", "insane", "ridiculous", "disgusting",
    "terrible", "awful", "worst", "selfish", "lazy", "jerk", "freak",
    "what's wrong with you", "your fault", "nobody cares"
]

# Preprocessing and label functions (unchanged)
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s\?\!]', ' ', text)
        return text
    return ""

def determine_label(row):
    if row['Manipulative'] == 0:
        return "Healthy"
    elif row['Manipulative'] == 1:
        if any(keyword in row['Dialogue'].lower() for keyword in TOXIC_INDICATORS):
            return "Toxic"
        return "Manipulative"
    return "Unknown"

# Feature extraction (unchanged)
def extract_speaker_features(text, speaker_label=None):
    features = []
    text_lower = text.lower()
    
    healthy_count = sum(1 for indicator in HEALTHY_INDICATORS if indicator in text_lower)
    manipulative_count = sum(1 for indicator in MANIPULATIVE_INDICATORS if indicator in text_lower)
    toxic_count = sum(1 for indicator in TOXIC_INDICATORS if indicator in text_lower)
    
    word_count = len(text_lower.split())
    if word_count > 0:
        healthy_ratio = healthy_count / word_count * 100
        manipulative_ratio = manipulative_count / word_count * 100
        toxic_ratio = toxic_count / word_count * 100
    else:
        healthy_ratio = manipulative_ratio = toxic_ratio = 0
    
    features.extend([healthy_count, manipulative_count, toxic_count, healthy_ratio, manipulative_ratio, toxic_ratio])
    features.append(text_lower.count('?'))
    features.append(text_lower.count('!'))
    features.append(sum(1 for word in text_lower.split() if word.isupper()))
    features.append(len(text_lower))
    
    victim_indicators = sum(1 for phrase in ["sorry", "my fault", "please don’t", "i’ll try harder"] if phrase in text_lower)
    features.append(victim_indicators)
    
    return features

class RelationshipChatClassifier:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.model = None
        self.label_encoder = None
        self.scaler = None
    
    def preprocess_data(self, data_path):
        df = pd.read_csv(data_path)
        df['Label'] = df.apply(determine_label, axis=1)
        df['Processed_Dialogue'] = df['Dialogue'].apply(preprocess_text)
        print("Label Distribution:")
        print(df['Label'].value_counts())
        return df
    
    def extract_all_features(self, texts):
        return np.array([extract_speaker_features(text) for text in texts])
    
    def train(self, data_path):
        df = self.preprocess_data(data_path)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 3))
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['Label'])
        
        X_tfidf = self.tfidf_vectorizer.fit_transform(df['Processed_Dialogue'])
        X_extra = self.extract_all_features(df['Processed_Dialogue'])
        self.scaler = StandardScaler()
        X_extra_scaled = self.scaler.fit_transform(X_extra)
        X_combined = np.hstack((X_tfidf.toarray(), X_extra_scaled))
        
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.1, random_state=12, stratify=y)
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, class_weight='balanced', random_state=42)
        self.model.fit(X_train_resampled, y_train_resampled)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        return accuracy
    
    def split_conversation(self, text):
        lines = text.strip().split('\n')
        speakers = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'^([^:]+):(.+)$', line)  # Match anything before ':' as speaker
            if match:
                speaker, content = match.groups()
                speaker = speaker.strip()
                content = content.strip()
                if speaker not in speakers:
                    speakers[speaker] = []
                speakers[speaker].append(content)
            else:
                # If no colon, treat as continuation of last speaker (unlikely in your case)
                if speakers and list(speakers.keys())[-1]:
                    speakers[list(speakers.keys())[-1]].append(line)
                else:
                    speakers["Unknown"] = [line]
        
        return speakers
    
    def predict(self, text):
        processed_text = preprocess_text(text)
        speakers = self.split_conversation(text)
        
        # Overall prediction
        X_tfidf = self.tfidf_vectorizer.transform([processed_text])
        X_extra = self.extract_all_features([processed_text])
        X_extra_scaled = self.scaler.transform(X_extra)
        X_combined = np.hstack((X_tfidf.toarray(), X_extra_scaled))
        
        y_pred = self.model.predict(X_combined)
        label = self.label_encoder.inverse_transform(y_pred)[0]
        probabilities = self.model.predict_proba(X_combined)[0]
        confidence = max(probabilities)
        
        # Rule-based corrections
        text_lower = processed_text.lower()
        healthy_indicators = sum(1 for indicator in HEALTHY_INDICATORS if indicator in text_lower)
        manipulative_indicators = sum(1 for indicator in MANIPULATIVE_INDICATORS if indicator in text_lower)
        toxic_indicators = sum(1 for indicator in TOXIC_INDICATORS if indicator in text_lower)
        
        if healthy_indicators >= 3 and manipulative_indicators <= 1 and toxic_indicators == 0:
            label = "Healthy"
            confidence = max(confidence, 0.9)
        elif label == "Toxic" and toxic_indicators < 2 and manipulative_indicators > toxic_indicators:
            label = "Manipulative"
            confidence = max(confidence, 0.85)
        
        # Speaker analysis for Manipulative/Toxic
        speaker_scores = {}
        victim = None
        culprit = None
        
        if label in ["Manipulative", "Toxic"]:
            for speaker, lines in speakers.items():
                speaker_text = " ".join(lines)
                X_tfidf_sp = self.tfidf_vectorizer.transform([speaker_text])
                X_extra_sp = self.extract_all_features([speaker_text])
                X_extra_scaled_sp = self.scaler.transform(X_extra_sp)
                X_combined_sp = np.hstack((X_tfidf_sp.toarray(), X_extra_scaled_sp))
                
                sp_label = self.label_encoder.inverse_transform(self.model.predict(X_combined_sp))[0]
                sp_probs = self.model.predict_proba(X_combined_sp)[0]
                
                manip_score = sp_probs[self.label_encoder.transform(["Manipulative"])[0]]
                toxic_score = sp_probs[self.label_encoder.transform(["Toxic"])[0]]
                victim_score = sum(1 for phrase in ["sorry", "my fault", "please don’t", "i’ll try harder"] if phrase in speaker_text.lower())
                
                speaker_scores[speaker] = {
                    "label": sp_label,
                    "manip_score": manip_score,
                    "toxic_score": toxic_score,
                    "victim_score": victim_score
                }
            
            manip_toxic_scores = {sp: max(scores["manip_score"], scores["toxic_score"]) for sp, scores in speaker_scores.items()}
            victim_scores = {sp: scores["victim_score"] for sp, scores in speaker_scores.items()}
            
            culprit = max(manip_toxic_scores, key=manip_toxic_scores.get)
            victim = max(victim_scores, key=victim_scores.get) if max(victim_scores.values()) > 0 else None
            
            if victim == culprit:
                victim = min(manip_toxic_scores, key=manip_toxic_scores.get) if len(speakers) > 1 else None
        
        # Generate separate advice for culprit and victim
        advice = self.generate_advice(label, victim, culprit)
        techniques, vulnerabilities = self.extract_concerns(text, label)
        
        return {
            'label': label,
            'confidence': confidence,
            'advice': advice,  # Now a dict with 'general', 'culprit', 'victim'
            'techniques': techniques,
            'vulnerabilities': vulnerabilities,
            'victim': victim if label in ["Manipulative", "Toxic"] else None,
            'culprit': culprit if label in ["Manipulative", "Toxic"] else None,
            'speaker_analysis': speaker_scores if label in ["Manipulative", "Toxic"] else {},
            'debug_info': {
                'healthy_indicators': healthy_indicators,
                'manipulative_indicators': manipulative_indicators,
                'toxic_indicators': toxic_indicators
            }
        }
    
    def generate_advice(self, label, victim=None, culprit=None):
        advice_mapping = {
    "Healthy": {
        "general": [
            "Let me play a song for you guys: 'Malare Ninne.mp3'\n\n1. Nurture What Works\nCommunication flowing? Trust solid? Keep investing in these strengths.\n\n2. Stay Present\nAppreciate what you have instead of worrying about what could go wrong.\n\n3. Regular Check-ins\nEven great relationships need maintenance. Talk honestly about needs and feelings.",
            
            "The perfect song for this kind of relationship would be 'Vazhithunaiye.mp3'\n\n1. Celebrate Balance\nEqual give and take is rare—protect it through mutual respect.\n\n2. Grow Together\nHealthy partners encourage each other's individual growth.\n\n3. Small Things Matter\nContinue noticing and appreciating the little gestures that strengthen your bond.",
            
            "Feels like that bike ride from adiye raasali raasali ey nillu,breezy la...\n\n1. Trust Your Peace\nDrama-free doesn't mean boring—it means secure.\n\n2. Expand Your Connection\nExplore new experiences together to keep the relationship fresh.\n\n3. Share Your Model\nYour healthy relationship can inspire others—be open about what works."
        ]
    },
    "Manipulative": {
        "culprit": [
            "Ivlo neram nee romba nallavan nu nenachen\n\n1. Come Clean\nIf you're playing games to control the vibe, stop—honesty feels way better.\n\n2. Dig Deep\nWhy do you need to pull the strings? Feeling insecure? Face that first.\n\n3. Let Them Choose\nTheir decisions aren't yours to steer—respect what they want, no tricks.",
            
            "Ennada romba bayangarmana aala iruka!\n\n1. Own It\nIf you're dangling carrots to keep them hooked, admit it—change starts now.\n\n2. Talk Clear\nNo more vague plans or half-truths—just say what you mean.\n\n3. Check Your Habits\nAsk a friend if you come off controlling—they'll keep it real with you.",
            
            "Irukku na irukku nu sollu illa na illa nu sollu!\n\n1. Spot Your Moves\nNotice when you're shutting down to get your way—awareness is key.\n\n2. Loosen the Grip\nYou don't need to control everything—let things flow naturally.\n\n3. Practice Openness\nTry sharing your feelings instead of icing them out—it's a game-changer."
        ],
        
        "victim": [
            "Unaku venum da, ithuvum venum ithukum melayum venum, seri seri kanna thoda!\n\n1. Trust Your Instincts\nIf their mixed signals feel off, they probably are—don't second-guess yourself.\n\n2. Set Your Limits\nTell them what's not cool, like dodging your questions, and stick to it.\n\n3. Phone a Friend\nVent to someone you trust—they'll help you untangle the confusion.",
            
            "Puriyudha? You are that Banerjee from 'Aadhavan,' time to shine like Pitchumani!\n\n1. Track the Signs\nNotice if they twist your words to make you feel bad—write it down to stay clear.\n\n2. Protect Your Space\nIf their drama's draining, take a breather—you're allowed to step back.\n\n3. Hold Your Ground\nYou don't owe them endless chances—you're worth straight-up respect.",
            
            "Grease dappa va epdi vodhacha?\n\n1. Name the Game\nIf they're jerking you around, call it out—you don't have to play along.\n\n2. Find Your Crew\nLean on people who've got your back—they'll remind you what's normal.\n\n3. Focus on You\nIf this keeps up, think about what keeps you happy—your peace comes first."
        ]
    },
    
    "Toxic": {
        "culprit": [
            "Ennada romba bayangarmana aala iruka!\n\n1. Take Responsibility\nIf your temper's hurting them, own it—blaming them won't fix anything.\n\n2. Hit Pause\nFeeling heated? Walk away for a sec—cool heads make better choices.\n\n3. Get Help\nTalk to someone wise, like a counselor—they'll show you healthier ways.",
            
            "Dei ivlo naal en kudave irundu ella velayum paathuruka\n\n1. Know Your Sparks\nFigure out what sets you off—tiredness, stress?—and plan to stay calm.\n\n2. Fix It Right\nMessed up? Say sorry and back it up with better actions next time.\n\n3. Learn to Listen\nHear them out without jumping in—you'll fight less and connect more.",
            
            "Ivlo neram nee romba nallavan nu nenachen!\n\n1. Stop the Blame\nIf you're punishing them with silence, that's on you—take charge of your mood.\n\n2. Give Space, Not Pain\nNeed a break? Say so kindly—don't leave them hanging.\n\n3. Build Better Habits\nTry talking it out or writing it down—there are ways to cope that don't hurt."
        ],
        
        "victim": [
            "When you thought you were Sachein Vijay but it turns out you're Kaadhal Bharath\n\n1. Put You First\nIf their outbursts kill your vibe, it's okay to prioritize your calm.\n\n2. Draw a Line\nSay what you won't take, like shouting, and mean it—no excuses.\n\n3. Grab Support\nCall a friend or hotline—you don't have to handle this on your own.",
            
            "Maruvaarthai padalam nu ponna ippo ellu vaaya pookala pottu vittangala\n\n1. See It Straight\nIf they're always flipping the script, that's not fair—trust what you feel.\n\n2. Take Time Out\nNeed a moment? Step away to clear your head—you're allowed.\n\n3. Know You're Enough\nYou don't deserve this chaos—you're worth someone who cares right.",
            
            "Enna sonna butterflies in the stomach aa!\n\n1. Stay Steady\nIf they're tearing you down, remind yourself what's true—your worth doesn't budge.\n\n2. Talk to Someone\nA friend or pro can help you see options—don't stay stuck alone.\n\n3. Plan for Better\nThink about what makes you smile—you deserve a life that feels good."
        ]
    }
}
        
        advice_dict = advice_mapping.get(label, {"general": ["Unable to provide specific advice."]})
        if label in ["Manipulative", "Toxic"]:
            return {
                "general": advice_dict.get("general", ["Check the vibe, something’s off da!"])[0],
                "culprit": advice_dict["culprit"][np.random.randint(len(advice_dict["culprit"]))],  # Random to avoid repetition
                "victim": advice_dict["victim"][np.random.randint(len(advice_dict["victim"]))]      # Random to avoid repetition
            }
        return {"general": advice_dict["general"][np.random.randint(len(advice_dict["general"]))], "culprit": None, "victim": None}
    
    def extract_concerns(self, text, label):
        techniques = []
        vulnerabilities = []
        text_lower = text.lower()
        
        if label in ["Manipulative", "Toxic"]:
            if "you always" in text_lower or "you never" in text_lower:
                techniques.append("Overgeneralization")
            if "if you loved me" in text_lower or "if you cared" in text_lower:
                techniques.append("Emotional Manipulation")
            if "everyone thinks" in text_lower or "nobody would" in text_lower:
                techniques.append("Social Pressure")
            if "you're imagining" in text_lower or "remember when you" in text_lower:
                techniques.append("Gaslighting")
            if "i'll leave" in text_lower or "i can find someone" in text_lower:
                techniques.append("Threats/Ultimatums")
            if "after all i've done" in text_lower or "i sacrificed" in text_lower:
                techniques.append("Guilt-Tripping")
            if text_lower.count("you") > 10:
                techniques.append("Blame-Shifting")
                
            apology_count = text_lower.count("sorry") + text_lower.count("my fault")
            if apology_count >= 2:
                vulnerabilities.append("Excessive Apologizing")
            
            if "i need you" in text_lower or "can't live without" in text_lower or "don’t leave me" in text_lower:
                vulnerabilities.append("Dependency")
            
            if "i'm not good enough" in text_lower or "i'm worthless" in text_lower or "i’m a failure" in text_lower:
                vulnerabilities.append("Low Self-Esteem")
            
            if "please don’t" in text_lower or "i’ll try harder" in text_lower or "i’ll do anything" in text_lower:
                plea_count = sum(1 for phrase in ["please don’t", "i’ll try harder", "i’ll do anything"] if phrase in text_lower)
                if plea_count >= 2:
                    vulnerabilities.append("People-Pleasing")
            
            if "i don’t know what to do" in text_lower or "i’m confused" in text_lower or "help me decide" in text_lower:
                vulnerabilities.append("Indecisiveness")
            
            if not vulnerabilities and label in ["Manipulative", "Toxic"]:
                vulnerabilities.append("Emotional Vulnerability")
        
        return techniques, vulnerabilities
    
    def save_model(self, filename="relationship_chat_model.pkl"):
        model_components = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_components, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename="relationship_chat_model.pkl"):
        with open(filename, 'rb') as f:
            model_components = pickle.load(f)
        self.tfidf_vectorizer = model_components['tfidf_vectorizer']
        self.model = model_components['model']
        self.label_encoder = model_components['label_encoder']
        self.scaler = model_components['scaler']
        print(f"Model loaded from {filename}")

def run_streamlit_app():
    st.title("Relationship Chat Analyzer")
    st.write("Upload or paste a chat to analyze dynamics and identify victim/culprit.")
    
    classifier = RelationshipChatClassifier()
    show_debug = st.sidebar.checkbox("Show debugging info", value=False)
    
    try:
        classifier.load_model()
        st.success("Model loaded successfully!")
    except:
        st.warning("Trained model not found. Please train the model first.")
        if st.button("Train Model"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    accuracy = classifier.train("mentalmanip_con.csv")
                    classifier.save_model()
                    st.success(f"Model trained with {accuracy:.2%} accuracy!")
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    st.info("Ensure 'mentalmanip_con.csv' is in the same directory.")
    
    uploaded_file = st.file_uploader("Upload chat file", type="txt")
    use_text_input = st.checkbox("Or paste chat text directly")
    chat_text = st.text_area("Paste chat conversation here:", height=200) if use_text_input else uploaded_file.getvalue().decode("utf-8") if uploaded_file else ""
    
    if chat_text:
        st.subheader("Chat Preview")
        st.text_area("Conversation", chat_text, height=200, key="preview")
        
        if st.button("Analyze Chat"):
            with st.spinner("Analyzing..."):
                result = classifier.predict(chat_text)
                
                st.subheader("Analysis Results")
                label = result['label']
                if label == "Healthy":
                    st.success(f"Prediction: {label} (Confidence: {result['confidence']:.2%})")
                elif label == "Manipulative":
                    st.warning(f"Prediction: {label} (Confidence: {result['confidence']:.2%})")
                else:
                    st.error(f"Prediction: {label} (Confidence: {result['confidence']:.2%})")
                
                st.subheader("Relationship Advice")
                st.write(f"**General Advice**: {result['advice']['general']}")
                
                if label in ["Manipulative", "Toxic"]:
                    st.subheader("Victim and Culprit")
                    st.write(f"**Culprit**: {result['culprit'] or 'Unknown'}")
                    st.write(f"**Advice for {result['culprit'] or 'Culprit'}**: {result['advice']['culprit']}")
                    st.write(f"**Victim**: {result['victim'] or 'Unknown'}")
                    st.write(f"**Advice for {result['victim'] or 'Victim'}**: {result['advice']['victim']}")
                
                with st.expander("See More Details"):
                    if result['techniques']:
                        st.subheader("Potential Manipulation Techniques")
                        for technique in result['techniques']:
                            st.write(f"- {technique}")
                    if result['vulnerabilities']:
                        st.subheader("Potential Vulnerabilities")
                        for vulnerability in result['vulnerabilities']:
                            st.write(f"- {vulnerability}")
                    if result['speaker_analysis']:
                        st.subheader("Speaker Analysis")
                        st.write(result['speaker_analysis'])
                
                if show_debug:
                    st.subheader("Debug Information")
                    st.write(f"Healthy indicators: {result['debug_info']['healthy_indicators']}")
                    st.write(f"Manipulative indicators: {result['debug_info']['manipulative_indicators']}")
                    st.write(f"Toxic indicators: {result['debug_info']['toxic_indicators']}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        print("Training model...")
        classifier = RelationshipChatClassifier()
        classifier.train("mentalmanip_con.csv")
        classifier.save_model()
        print("Model training complete!")
    else:
        run_streamlit_app()