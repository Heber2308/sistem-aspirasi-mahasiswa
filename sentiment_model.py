import re
import string
import numpy as np
import pickle
import os
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

class SentimentAnalyzer:
    """Kelas untuk analisis sentimen menggunakan Naïve Bayes dengan Negation Handling"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        self.classifier = MultinomialNB(alpha=0.1)
        self.feature_selector = None
        self.stemmer = StemmerFactory().create_stemmer()
        self.stop_words = set(stopwords.words('indonesian'))
        self.classes_ = None
        
        # ========== KATA NEGASI ==========
        self.negation_words = {
            'tidak', 'bukan', 'gak', 'nggak', 'ngga', 'ga', 'tak',
            'jangan', 'belum', 'blm', 'kurang', 'tdk', 'td',
            'tanpa', 'tiada', 'jgn'
        }
        
        # ========== KATA SENTIMEN NEGATIF ==========
        self.negative_sentiment_words = {
            'buruk', 'jelek', 'berantakan', 'rusak', 'error', 'lemot',
            'sulit', 'susah', 'ribet', 'lambat', 'lamban', 'kotor',
            'bising', 'panas', 'pengap', 'sempit', 'kumuh', 'bau',
            'kecewa', 'mengecewakan', 'parah', 'payah', 'aneh',
            'jorok', 'kacau', 'kasar', 'mahal', 'malas', 'membosankan', 'mengerikan', 'menjengkelkan',
            'menyebalkan', 'stress', 'tua', 'bermasalah', 'jorok', 'joroknya',
            'nakal', 'norak', 'sampah', 'sombong', 'tolol', 'berantakan',
            'kampungan', 'murahan', 'cacat', 'capek', 'culas',
            'bego', 'bodoh', 'brengsek', 'busuk', 'cerewet',
        }
        
        # ========== KATA SENTIMEN POSITIF ==========
        self.positive_sentiment_words = {
            'bagus', 'baik', 'mantap', 'keren', 'oke', 'ok', 'sip',
            'cepat', 'mudah', 'responsif', 'ramah', 'nyaman', 'bersih',
            'lengkap', 'modern', 'inovatif', 'membantu', 'puas', 'memuaskan',
            'luar_biasa', 'sangat_bagus', 'sangat_baik', 'aman', 'asyik', 'asik',
            'bangga', 'benar', 'berhasil', 'cerdas', 'canggih', 'cemerlang',
            'efektif', 'efisien', 'elegan', 'enak', 'fantastis', 'favorit',
            'gratis', 'hebat', 'indah', 'istimewa', 'juara',
            'lancar', 'luas', 'manis', 'menarik', 'mengesankan',
            'profesional', 'rapi', 'stabil', 'sukses', 'super', 'terbaik',
            'terpercaya', 'top', 'transparan', 'berkualitas', 'berprestasi',
            'adil', 'sejuk', 'lega', 'puas', 'senang', 'gembira', 'ceria',
            'berguna', 'dermawan', 'detail', 'gokil', 'harmonis', 'hemat',
            'ideal', 'intuitif', 'kebanggaan', 'kece', 'kompeten',
            'legendaris', 'optimal', 'paten', 'pintar', 'populer',
            'rekomendasi', 'royal', 'sabar', 'sederhana', 'segar',
            'sempurna', 'sensasional', 'simpel', 'solid', 'solutif',
            'spesial', 'tanggap', 'tepat', 'terjamin', 'terkenal',
            'tersenyum', 'teruji', 'unggul', 'variatif', 'wangi',
            'kinclong', 'cekatan', 'gercep', 'sigap', 'cemerlang',
            'elite', 'gagah', 'megah', 'apik', 'tertata', 'lebar',
            'lapang', 'teduh', 'rindang', 'asri', 'hijau', 'sehat',
            'higienis', 'terawat', 'terpilah', 'terjangkau', 'murah',
            'ekonomis', 'worthit', 'berfaedah', 'nendang', 'nampol',
            'dahsyat', 'spektakuler', 'mantul', 'jos', 'maknyus',
            'ngangenin', 'bikinbetah', 'adem', 'ademayem', 'tentrem',
            'damai', 'sreg', 'klop', 'cocok', 'pas', 'sreg', 'plong',
            'lega', 'enteng', 'ringan', 'plong', 'tentram', 'amanah',
            'jujur', 'ikhlas', 'sabar', 'telaten', 'teliti', 'cermat',
            'detail', 'rapi', 'apik', 'resik'
        }
        
        self.normalization_dict = {
            "gak": "tidak", "ga": "tidak", "ngga": "tidak", "nggak": "tidak",
            "udah": "sudah", "dah": "sudah", "sdh": "sudah",
            "bgt": "sangat", "banget": "sangat",
            "tp": "tapi", "jg": "juga",
            "klo": "kalau", "kalo": "kalau",
            "yg": "yang", "dgn": "dengan",
            "aja": "saja",
            "sya": "saya", "gw": "saya", "gue": "saya",
            "lu": "kamu", "loe": "kamu",
            "tdk": "tidak", "td": "tidak",
            "blm": "belum",
            "jgn": "jangan",
            "brantakan": "berantakan",
            "amburadul": "berantakan",
            "kaga": "tidak",
            "g": "tidak",
            "ndak": "tidak",
            "nda": "tidak",
            "nd": "tidak",
            "kagak": "tidak",
            "enggak": "tidak",
            "engga": "tidak",
        }
    
    def _is_negative_word(self, word):
        return word.lower() in self.negative_sentiment_words
    
    def _is_positive_word(self, word):
        return word.lower() in self.positive_sentiment_words
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[@#][^\s]+', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)  # Hapus SEMUA tanda baca
        text = ' '.join(text.split())
        return text
    
    def handle_negation(self, text):
        """
        Menangani negasi DENGAN MEMAHAMI SENTIMEN KATA.
        - "tidak bagus" → "tidak TIDAK_bagus" (negasi + positif = negatif)
        - "tidak berantakan" → "tidak YA_berantakan" (negasi + negatif = positif)
        - "tidak jelek" → "tidak YA_jelek" (negasi + negatif = positif)
        """
        words = text.split()
        result = []
        negation_active = False
        negation_just_set = False
        
        for i, word in enumerate(words):
            # Cek double negation
            if word in self.negation_words and negation_active:
                negation_active = False
                result.append(word)
                continue
            
            if word in self.negation_words:
                negation_active = True
                negation_just_set = True
                result.append(word)
                continue
            
            if negation_active:
                if self._is_negative_word(word):
                    # Negasi + Negatif = POSITIF → prefix YA_
                    result.append(f"YA_{word}")
                elif self._is_positive_word(word):
                    # Negasi + Positif = NEGATIF → prefix TIDAK_
                    result.append(f"TIDAK_{word}")
                else:
                    # Kata netral → tetap prefix TIDAK_
                    result.append(f"TIDAK_{word}")
                negation_active = False
            else:
                result.append(word)
        
        return ' '.join(result)
    
    def normalize_text(self, text):
        words = text.split()
        normalized_words = []
        for word in words:
            if word.startswith('TIDAK_'):
                original = word[6:]
                normalized = self.normalization_dict.get(original, original)
                normalized_words.append(f"TIDAK_{normalized}")
            elif word.startswith('YA_'):
                original = word[3:]
                normalized = self.normalization_dict.get(original, original)
                normalized_words.append(f"YA_{normalized}")
            else:
                normalized_words.append(self.normalization_dict.get(word, word))
        return ' '.join(normalized_words)
    
    def remove_stopwords(self, text):
        words = text.split()
        filtered_words = []
        for word in words:
            if word in self.negation_words:
                filtered_words.append(word)
            elif word.startswith('TIDAK_') or word.startswith('YA_'):
                filtered_words.append(word)
            elif word not in self.stop_words:
                filtered_words.append(word)
        return ' '.join(filtered_words)
    
    def stem_text(self, text):
        words = text.split()
        stemmed_words = []
        for word in words:
            if word in self.negation_words:
                stemmed_words.append(word)
            elif word.startswith('TIDAK_'):
                original_word = word[6:]
                stemmed_word = self.stemmer.stem(original_word)
                stemmed_words.append(f"TIDAK_{stemmed_word}")
            elif word.startswith('YA_'):
                original_word = word[3:]
                stemmed_word = self.stemmer.stem(original_word)
                stemmed_words.append(f"YA_{stemmed_word}")
            else:
                stemmed_words.append(self.stemmer.stem(word))
        return ' '.join(stemmed_words)
    
    def preprocess(self, text):
        text = self.clean_text(text)
        text = self.normalize_text(text)
        text = self.handle_negation(text)
        text = self.remove_stopwords(text)
        text = self.stem_text(text)
        return text
    
    def train(self, texts, labels):
        # ===== PENTING: Preprocess SEMUA data training =====
        print("📝 Memproses data training dengan negation handling...")
        processed_texts = [self.preprocess(text) for text in texts]
        
        # Debug: lihat contoh hasil preprocessing
        print("\n🔍 Contoh hasil preprocessing:")
        for i in range(min(5, len(processed_texts))):
            print(f"  '{texts[i]}' → '{processed_texts[i]}' ({labels[i]})")
        print()
        
        X = self.vectorizer.fit_transform(processed_texts)
        y = np.array(labels)
        
        self.classes_ = np.unique(y)
        
        print("🔍 Melakukan feature selection...")
        self.feature_selector = SelectKBest(chi2, k=min(2000, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("🤖 Melatih Multinomial Naive Bayes...")
        self.classifier.fit(X_train, y_train)
        
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("📊 Cross-validation (5-fold)...")
        cv_scores = cross_val_score(self.classifier, X_selected, y, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"✅ Akurasi: {accuracy*100:.2f}%")
        print(f"📈 CV Accuracy: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict(self, text):
        processed_text = self.preprocess(text)
        X = self.vectorizer.transform([processed_text])
        X_selected = self.feature_selector.transform(X)
        
        sentiment = self.classifier.predict(X_selected)[0]
        probabilities = self.classifier.predict_proba(X_selected)[0]
        
        probabilities_dict = {}
        for i, class_label in enumerate(self.classifier.classes_):
            probabilities_dict[class_label] = round(probabilities[i] * 100, 2)
        
        confidence = max(probabilities_dict.values())
        
        return {
            'sentiment': sentiment,
            'confidence': round(confidence, 2),
            'probabilities': {
                'positif': probabilities_dict.get('positif', 0),
                'netral': probabilities_dict.get('netral', 0),
                'negatif': probabilities_dict.get('negatif', 0)
            }
        }
    
    def debug_preprocess(self, text):
        """Menampilkan langkah-langkah preprocessing"""
        print(f"📝 Original: '{text}'")
        cleaned = self.clean_text(text)
        print(f"🧹 Cleaned: '{cleaned}'")
        normalized = self.normalize_text(cleaned)
        print(f"📖 Normalized: '{normalized}'")
        negated = self.handle_negation(normalized)
        print(f"🚫 After Negation: '{negated}'")
        stopwords_removed = self.remove_stopwords(negated)
        print(f"🛑 Stopwords Removed: '{stopwords_removed}'")
        stemmed = self.stem_text(stopwords_removed)
        print(f"🌱 Stemmed: '{stemmed}'")
        return stemmed
    
    def test_negation(self):
        """Test kasus negasi"""
        test_cases = [
            ("tidak bagus", "negatif"),
            ("tidak berantakan", "positif"),
            ("kelas tidak berantakan", "positif"),
            ("tidak jelek", "positif"),
            ("tidak buruk", "positif"),
            ("tidak rapi", "positif"),  # rapi = positif
            ("tidak ramah", "negatif"),  # ramah = positif
            ("bagus", "positif"),
            ("berantakan", "negatif"),
            ("tidak kotor", "positif"),
            ("dosen tidak ramah", "negatif"),
            ("satpam tidak ramah", "negatif"),
            ("pelayanan tidak bagus", "negatif"),
            ("fasilitas tidak buruk", "positif"),
            ("ruangan tidak berantakan", "positif"),
            ("kamar mandi tidak kotor", "positif"),
            ("krs tidak jelek", "positif"),
        ]
        
        print("\n" + "="*80)
        print("🧪 TEST NEGATION CASES")
        print("="*80)
        correct = 0
        for text, expected in test_cases:
            result = self.predict(text)
            status = "✅" if result['sentiment'] == expected else "❌"
            if status == "✅":
                correct += 1
            print(f"{status} '{text}' → {result['sentiment']} (expected: {expected})")
        
        print(f"\n📊 Akurasi Test Negasi: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")
    
    def save_model(self, path='model/naive_bayes.pkl'):
        os.makedirs('model', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'feature_selector': self.feature_selector,
                'classes_': self.classes_
            }, f)
        print(f"✅ Model disimpan ke {path}")
    
    def load_model(self, path='model/naive_bayes.pkl'):
        with open(path, 'rb') as f:
            saved = pickle.load(f)
            self.vectorizer = saved['vectorizer']
            self.classifier = saved['classifier']
            self.feature_selector = saved.get('feature_selector')
            self.classes_ = saved.get('classes_')
        print(f"✅ Model dimuat dari {path}")
    
    def generate_suggestions(self, text, sentiment, category):
        # ... (keep your existing suggestions code, unchanged) ...
        suggestions_dict = {
            'akademik': {
                'positif': ['Pertahankan kualitas pembelajaran...'],
                'negatif': ['Perlu evaluasi kurikulum...'],
                'netral': ['Evaluasi berkala...']
            },
            # ... (rest of your suggestions dict)
        }
        # simplified fallback
        return {
            'suggestions': ['Saran untuk ' + sentiment],
            'category': category,
            'sentiment': sentiment
        }


# ========== TRAINING DATA MANAGEMENT ==========

def load_training_data_from_csv(csv_path='training_data.csv'):
    texts = []
    labels = []
    
    try:
        if os.path.exists(csv_path):
            print(f"📂 Membaca file CSV: {csv_path}")
            df = pd.read_csv(csv_path, encoding='utf-8')
            texts = df['text'].tolist()
            labels = df['label'].tolist()
            print(f"✅ Loaded {len(texts)} training samples dari CSV")
            
            # Count distributions
            unique, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique, counts):
                print(f"   - {label}: {count}")
            
            return texts, labels
        else:
            print(f"⚠️  File {csv_path} tidak ditemukan.")
            return [], []
    except Exception as e:
        print(f"❌ Error membaca CSV: {e}")
        return [], []


# Inisialisasi model global
sentiment_analyzer = SentimentAnalyzer()


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Load data dari CSV
    print("="*80)
    print("🚀 SENTIMENT ANALYZER DENGAN NEGATION HANDLING")
    print("="*80)
    
    texts, labels = load_training_data_from_csv('training_data.csv')
    
    if texts and labels:
        # Train model
        sentiment_analyzer.train(texts, labels)
        
        # Test kasus negasi
        print("\n🔍 Testing specific negation cases...")
        sentiment_analyzer.test_negation()
        
        # Debug preprocessing untuk kasus spesifik
        print("\n🔍 Debug preprocessing:")
        sentiment_analyzer.debug_preprocess("kelas tidak berantakan")
        print()
        sentiment_analyzer.debug_preprocess("tidak bagus")
        
        # Save model
        sentiment_analyzer.save_model()
        
        # Interactive testing
        print("\n" + "="*80)
        print("💬 Masukkan teks untuk dianalisis (ketik 'exit' untuk keluar)")
        print("="*80)
        while True:
            user_input = input("\n📝 Teks: ")
            if user_input.lower() == 'exit':
                break
            if user_input.lower() == 'debug':
                debug_input = input("Teks untuk debug: ")
                sentiment_analyzer.debug_preprocess(debug_input)
                continue
            result = sentiment_analyzer.predict(user_input)
            print(f"📊 Sentimen: {result['sentiment']}")
            print(f"📈 Confidence: {result['confidence']}%")
            print(f"📊 Probabilitas: {result['probabilities']}")
    else:
        print("❌ Tidak ada data training. Pastikan file CSV tersedia.")
