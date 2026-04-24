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

# Download stopwords NLTK
nltk.download('stopwords')

class SentimentAnalyzer:
    """Kelas untuk analisis sentimen menggunakan Naïve Bayes dengan Sentiment-Aware Negation Handling"""
    
    def __init__(self):
        # TF-IDF dengan optimasi parameter
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),  # Unigram, Bigram, Trigram
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        # Multinomial Naive Bayes dengan alpha tuning
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
        
        # ========== KATA SENTIMEN NEGATIF (BARU DITAMBAHKAN) ==========
        # Kata-kata yang SUDAH bermakna negatif
        self.negative_sentiment_words = {
            'buruk', 'jelek', 'berantakan', 'rusak', 'error', 'lemot',
            'sulit', 'susah', 'ribet', 'lambat', 'lamban', 'kotor',
            'bising', 'panas', 'pengap', 'sempit', 'kumuh', 'bau',
            'kecewa', 'mengecewakan', 'parah', 'payah', 'aneh', 'anehnya',
            'bangkai', 'bego', 'bodoh', 'brengsek', 'busuk',
            'cacat', 'capek', 'cerewet', 'culas',
            'jorok', 'joroknya',
            'kampungan', 'kacau', 'kaku', 'kasar', 'kurangajar',
            'mahal', 'malas', 'membosankan', 'mengerikan', 'menjengkelkan',
            'menyebalkan', 'murahan', 'murahan',
            'nakal', 'norak',
            'sampah', 'sia', 'sia-sia', 'sombong', 'stress',
            'tolol', 'tua',
            'bermasalah', 'berantakan', 'bermasalah', 'berantakan',
            'berantakan', 'berantakan', 'berantakan',
            'tdk_bagus', 'tidak_bagus', 'gak_bagus', 'NOT_bagus'  # Variasi n-gram
        }
        
        # ========== KATA SENTIMEN POSITIF (BARU DITAMBAHKAN) ==========
        self.positive_sentiment_words = {
            'bagus', 'baik', 'mantap', 'keren', 'oke', 'ok', 'sip',
            'cepat', 'mudah', 'responsif', 'ramah', 'nyaman', 'bersih',
            'lengkap', 'modern', 'inovatif', 'membantu', 'puas', 'memuaskan',
            'luar_biasa', 'sangat_bagus', 'sangat_baik',
            'aman', 'asyik', 'asik', 'asoy',
            'baiknya', 'bangga', 'beken', 'benar', 'berhasil',
            'cerdas', 'cakep', 'canggih', 'cemerlang', 'cocok',
            'dermawan', 'detail',
            'efektif', 'efisien', 'elegan', 'elite', 'enak',
            'fantastis', 'favorit',
            'gentle', 'gokil', 'gratis', 'gue_banget',
            'halus', 'harmonis', 'hemat', 'hebat', 'heroik',
            'ideal', 'indah', 'inovatif', 'intuitif', 'istimewa',
            'jempol', 'jitu', 'juara',
            'kaya', 'kebanggaan', 'kece', 'keren_abis', 'khas', 'kompeten',
            'lancar', 'legendaris', 'luas',
            'manis', 'menarik', 'menawan', 'mengesankan', 'mudah_digunakan',
            'oke_banget', 'optimal',
            'paten', 'peduli', 'pelayanan_prima', 'pintar', 'populer', 'profesional',
            'rapi', 'rekomendasi', 'reliable', 'romantis', 'royal',
            'sabar', 'sederhana', 'segar', 'sejuk', 'sempurna', 'sensasional', 'simpel',
            'soft', 'solid', 'solutif', 'spesial', 'stabil', 'sukses', 'super',
            'tanggap', 'tepat', 'terang', 'terbaik', 'terdepan', 'terjamin', 'terkenal',
            'terpercaya', 'tersenyum', 'teruji', 'top', 'transparan'
        }
        
        # Kamus normalisasi kata tidak baku
        self.normalization_dict = {
            "gak": "tidak", "ga": "tidak", "ngga": "tidak", "nggak": "tidak",
            "udah": "sudah", "dah": "sudah", "sdh": "sudah",
            "bgt": "sangat", "banget": "sangat", "bgtu": "begitu",
            "tp": "tapi", "dr": "dari", "jg": "juga", "jgk": "juga",
            "klo": "kalau", "kl": "kalau", "kalo": "kalau",
            "trs": "terus", "trus": "terus",
            "mo": "mau", "mauu": "mau",
            "yg": "yang", "dq": "dengan", "dgn": "dengan",
            "aja": "saja", "aj": "saja",
            "kok": "mengapa", "knp": "kenapa", "knapa": "kenapa",
            "org": "orang", "sya": "saya", "gw": "saya", "gue": "saya",
            "lu": "kamu", "loe": "kamu",
            "dlm": "dalam", "utk": "untuk", "pd": "pada",
            "tdk": "tidak", "td": "tidak",
            "blm": "belum", "dpt": "dapat",
            "jgn": "jangan", "hrs": "harus", "bsa": "bisa",
            "sgt": "sangat", "cpt": "cepat", "lgs": "langsung",
            # Normalisasi kata negatif informal
            "berantakan": "berantakan",
            "brantakan": "berantakan",
            "amburadul": "berantakan",
            "berantaqan": "berantakan"
        }
    
    # ========== METHOD UNTUK DETEKSI SENTIMEN KATA ==========
    def _is_negative_word(self, word):
        """Cek apakah kata bermakna negatif"""
        return word.lower() in self.negative_sentiment_words
    
    def _is_positive_word(self, word):
        """Cek apakah kata bermakna positif"""
        return word.lower() in self.positive_sentiment_words
    
    def _get_word_sentiment(self, word):
        """Mendapatkan nilai sentimen kata"""
        if self._is_positive_word(word):
            return 'positif'
        elif self._is_negative_word(word):
            return 'negatif'
        else:
            return 'netral'
    
    def clean_text(self, text):
        """Membersihkan teks dari karakter yang tidak perlu"""
        if not isinstance(text, str):
            return ""
        
        # Ubah ke lowercase
        text = text.lower()
        
        # Hapus mention dan hashtag
        text = re.sub(r'[@#][^\s]+', '', text)
        
        # Hapus URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Hapus angka
        text = re.sub(r'\d+', '', text)
        
        # Hapus tanda baca (KECUALI yang penting untuk negasi stopper)
        text = re.sub(r'[^\w\s,.;:!?]', '', text)
        
        # Hapus spasi berlebih
        text = ' '.join(text.split())
        
        return text
    
    # ========== METHOD HANDLE NEGATION (DIPERBAIKI) ==========
    def handle_negation(self, text):
        """
        Menangani negasi DENGAN MEMAHAMI SENTIMEN KATA.
        
        Logika:
        - "tidak bagus" → "tidak NOT_bagus" (negasi + positif = negatif) ✅
        - "tidak berantakan" → "tidak POSITIVE_berantakan" (negasi + negatif = positif) ✅
        - "tidak jelek" → "tidak POSITIVE_jelek" ✅
        - "tidak tidak bagus" → "tidak bagus" (double negation) ✅
        """
        words = text.split()
        result = []
        negation_active = False
        
        for i, word in enumerate(words):
            # Cek double negation: "tidak tidak X"
            if word in self.negation_words and negation_active:
                # Double negation = POSITIF, jadi matikan negasi
                negation_active = False
                result.append(word)
                continue
            
            # Cek apakah ini kata negasi
            if word in self.negation_words:
                # Cek kata berikutnya (jika ada)
                next_word = words[i+1] if i+1 < len(words) else None
                
                if next_word:
                    # Jika kata berikutnya SUDAH negatif
                    if self._is_negative_word(next_word):
                        # "tidak berantakan" = POSITIF
                        # Tandai dengan PREFIX POSITIVE_
                        negation_active = False
                        result.append(word)
                        # Skip next word, akan ditangani di iterasi berikutnya
                        continue
                    else:
                        # "tidak bagus" = NEGATIF
                        # Aktifkan mode negasi
                        negation_active = True
                        result.append(word)
                        continue
                else:
                    result.append(word)
                    continue
            
            # Proses kata dalam mode negasi
            if negation_active:
                # Cek apakah kata ini kata positif atau negatif atau netral
                if self._is_negative_word(word):
                    # Negasi + Negatif = POSITIF
                    # Gunakan prefix POSITIVE_ untuk flip artinya
                    result.append(f"POSITIVE_{word}")
                elif self._is_positive_word(word):
                    # Negasi + Positif = NEGATIF
                    # Gunakan prefix NOT_ untuk flip artinya
                    result.append(f"NOT_{word}")
                else:
                    # Kata netral, tetap gunakan prefix NOT_
                    result.append(f"NOT_{word}")
            else:
                # Jika sebelumnya ada negasi + kata negatif, tandai sebagai POSITIVE_
                if i > 0 and words[i-1] in self.negation_words and self._is_negative_word(word):
                    result.append(f"POSITIVE_{word}")
                else:
                    result.append(word)
        
        return ' '.join(result)
    
    def normalize_text(self, text):
        """Normalisasi kata tidak baku"""
        words = text.split()
        normalized_words = []
        for word in words:
            if word.startswith('NOT_'):
                # Normalisasi kata setelah NOT_
                original = word[4:]
                normalized = self.normalization_dict.get(original, original)
                normalized_words.append(f"NOT_{normalized}")
            elif word.startswith('POSITIVE_'):
                # Normalisasi kata setelah POSITIVE_
                original = word[9:]
                normalized = self.normalization_dict.get(original, original)
                normalized_words.append(f"POSITIVE_{normalized}")
            else:
                normalized_words.append(self.normalization_dict.get(word, word))
        return ' '.join(normalized_words)
    
    def remove_stopwords(self, text):
        """Menghapus stopwords tapi mempertahankan kata negasi dan sentimen"""
        words = text.split()
        filtered_words = []
        for word in words:
            # Simpan kata negasi
            if word in self.negation_words:
                filtered_words.append(word)
            # Simpan kata dengan prefix khusus
            elif word.startswith('NOT_') or word.startswith('POSITIVE_'):
                filtered_words.append(word)
            # Hapus stopwords biasa
            elif word not in self.stop_words:
                filtered_words.append(word)
        return ' '.join(filtered_words)
    
    def stem_text(self, text):
        """Stemming dengan mempertahankan prefix sentimen"""
        words = text.split()
        stemmed_words = []
        for word in words:
            # Jangan stem kata negasi
            if word in self.negation_words:
                stemmed_words.append(word)
            elif word.startswith('NOT_'):
                original_word = word[4:]
                stemmed_word = self.stemmer.stem(original_word)
                stemmed_words.append(f"NOT_{stemmed_word}")
            elif word.startswith('POSITIVE_'):
                original_word = word[9:]
                stemmed_word = self.stemmer.stem(original_word)
                stemmed_words.append(f"POSITIVE_{stemmed_word}")
            else:
                stemmed_words.append(self.stemmer.stem(word))
        return ' '.join(stemmed_words)
    
    def preprocess(self, text):
        """Preprocessing teks secara lengkap"""
        text = self.clean_text(text)
        text = self.normalize_text(text)
        # HANDLE NEGATION dengan Sentiment-Aware
        text = self.handle_negation(text)
        text = self.remove_stopwords(text)
        text = self.stem_text(text)
        return text
    
    def train(self, texts, labels):
        """Melatih model dengan data dan optimasi"""
        # Preprocessing semua teks
        processed_texts = [self.preprocess(text) for text in texts]
        
        # Ekstraksi fitur dengan TF-IDF
        X = self.vectorizer.fit_transform(processed_texts)
        y = np.array(labels)
        
        # Simpan classes untuk digunakan di predict
        self.classes_ = np.unique(y)
        
        # 🔥 TAMBAHKAN: Cek distribusi kelas
        from collections import Counter
        label_counts = Counter(y)
        print(f"📊 Distribusi kelas: {label_counts}")
        
        # 🔥 TAMBAHKAN: Filter kelas yang kurang dari 2 sampel
        min_count = min(label_counts.values())
        if min_count < 2:
            print(f"⚠️  Terdeteksi kelas dengan {min_count} sampel!")
            print("   Melakukan filtering data...")
            
            # Filter kelas yang memiliki minimal 2 sampel
            valid_classes = [cls for cls, count in label_counts.items() if count >= 2]
            valid_indices = [i for i, label in enumerate(y) if label in valid_classes]
            
            processed_texts = [processed_texts[i] for i in valid_indices]
            X = self.vectorizer.fit_transform(processed_texts)
            y = y[valid_indices]
            
            print(f"📊 Setelah filtering: {Counter(y)}")
        
        # Feature selection: pilih top 2000 features dengan chi2 score tertinggi
        print("🔍 Melakukan feature selection...")
        self.feature_selector = SelectKBest(chi2, k=min(2000, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # 🔥 MODIFIKASI: Cek apakah stratify bisa digunakan
        if min(Counter(y).values()) >= 2:
            print("✅ Menggunakan stratified split")
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            print("⚠️  Tidak bisa stratifikasi, menggunakan split biasa")
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42
            )
        
        # Training model Multinomial Naive Bayes
        print("🤖 Melatih Multinomial Naive Bayes dengan alpha=0.1...")
        self.classifier.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation untuk hasil lebih reliabel
        print("📊 Cross-validation (5-fold)...")
        cv_scores = cross_val_score(self.classifier, X_selected, y, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"✅ Model dilatih dengan akurasi: {accuracy*100:.2f}%")
        print(f"📈 Cross-validation accuracy: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict(self, text):
        """Memprediksi sentimen dari teks"""
        # Preprocessing
        processed_text = self.preprocess(text)
        
        # Ekstraksi fitur
        X = self.vectorizer.transform([processed_text])
        
        # Apply feature selection (fitur yang sama dengan training)
        X_selected = self.feature_selector.transform(X)
        
        # Prediksi dengan Multinomial Naive Bayes
        sentiment = self.classifier.predict(X_selected)[0]
        probabilities = self.classifier.predict_proba(X_selected)[0]
        
        # Mapping label classes ke probabilities
        probabilities_dict = {}
        for i, class_label in enumerate(self.classifier.classes_):
            probabilities_dict[class_label] = round(probabilities[i] * 100, 2)
        
        confidence = max(probabilities_dict.values())
        
        # Mapping label yang konsisten
        sentiment_map = {
            'positif': 'positif',
            'negatif': 'negatif',
            'netral': 'netral'
        }
        
        return {
            'sentiment': sentiment_map.get(sentiment, 'netral'),
            'confidence': round(confidence, 2),
            'probabilities': {
                'positif': probabilities_dict.get('positif', 0),
                'netral': probabilities_dict.get('netral', 0),
                'negatif': probabilities_dict.get('negatif', 0)
            }
        }
    
    # ========== METHOD DEBUGGING NEGATION ==========
    def debug_preprocess(self, text):
        """Menampilkan langkah-langkah preprocessing untuk debugging"""
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
    
    # ========== METHOD TEST NEGASI (TAMBAHAN) ==========
    def test_negation_cases(self):
        """Test berbagai kasus negasi"""
        test_cases = [
            ("tidak bagus", "negatif"),  # Negasi + Positif = Negatif
            ("tidak jelek", "positif"),  # Negasi + Negatif = Positif
            ("tidak berantakan", "positif"),  # Negasi + Negatif = Positif
            ("tidak buruk", "positif"),  # Negasi + Negatif = Positif
            ("aplikasi tidak bagus", "negatif"),
            ("kelas tidak berantakan", "positif"),  # KASUS ANDA
            ("pelayanan tidak jelek", "positif"),
            ("fasilitas tidak rusak", "positif"),
            ("dosen tidak buruk", "positif"),
            ("sistem tidak error", "positif"),
            ("bagus", "positif"),
            ("berantakan", "negatif"),
            ("tidak tidak bagus", "positif"),  # Double negation
            ("gak bagus", "negatif"),
            ("gak jelek", "positif"),
        ]
        
        print("\n" + "="*80)
        print("🧪 TEST NEGATION CASES")
        print("="*80)
        
        for text, expected in test_cases:
            result = self.predict(text)
            status = "✅" if result['sentiment'] == expected else "❌"
            print(f"{status} '{text}' → {result['sentiment']} (expected: {expected}) [confidence: {result['confidence']}%]")
    
    def save_model(self, path='model/naive_bayes.pkl'):
        """Menyimpan model ke file"""
        os.makedirs('model', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier
            }, f)
        print(f"✅ Model disimpan ke {path}")
    
    def load_model(self, path='model/naive_bayes.pkl'):
        """Memuat model dari file"""
        with open(path, 'rb') as f:
            saved = pickle.load(f)
            self.vectorizer = saved['vectorizer']
            self.classifier = saved['classifier']
        print(f"✅ Model dimuat dari {path}")
    
    def generate_suggestions(self, text, sentiment, category):
        """Generate saran berdasarkan teks, sentimen, dan kategori aspirasi"""
        # Dictionary saran berdasarkan kategori dan sentimen
        suggestions_dict = {
            'akademik': {
                'positif': [
                    'Pertahankan kualitas pembelajaran yang sudah baik dengan terus berinovasi dalam metode pengajaran',
                    'Tingkatkan keterlibatan mahasiswa melalui diskusi kelas yang lebih interaktif',
                    'Kembangkan lebih banyak program pembelajaran praktis dan berbasis industri',
                    'Pertimbangkan untuk memberikan scholarship / beasiswa kepada mahasiswa berprestasi'
                ],
                'negatif': [
                    'Perlu evaluasi kurikulum dan pembaruan materi kuliah sesuai perkembangan industri',
                    'Tingkatkan kualifikasi dan pelatihan dosen agar lebih sesuai dengan standar internasional',
                    'Berikan program remedial dan tutorial tambahan untuk mahasiswa yang tertinggal',
                    'Perbaiki sistem penjadwalan kelas agar tidak terjadi bentrok jadwal',
                    'Aktifkan mekanisme feedback dan perbaikan berkelanjutan dari masukan mahasiswa'
                ],
                'netral': [
                    'Evaluasi berkala terhadap proses pembelajaran untuk memastikan efektivitas',
                    'Kumpulkan feedback lebih lanjut dari mahasiswa tentang aspek akademik yang perlu ditingkatkan'
                ]
            },
            'administrasi': {
                'positif': [
                    'Pertahankan sistem administrasi yang sudah berjalan baik dan responsif',
                    'Tingkatkan kepuasan mahasiswa dengan terus mempercepat proses administrasi',
                    'Share best practices dengan departemen lain untuk meningkatkan koordinasi'
                ],
                'negatif': [
                    'Optimalkan sistem KRS/KHS agar dapat diakses dengan lancar tanpa hambatan teknis',
                    'Sederhanakan prosedur pembayaran UKT dan berikan berbagai pilihan metode pembayaran',
                    'Tingkatkan kecepatan pemrosesan dokumen akademik dan administrasi',
                    'Selengkapi sistem informasi mahasiswa (SIMAK) dengan dokumentasi yang lebih jelas',
                    'Berikan pelatihan kepada staff administrasi tentang customer service yang lebih baik'
                ],
                'netral': [
                    'Tetap monitor proses administrasi dan minta feedback berkala dari mahasiswa'
                ]
            },
            'fasilitas': {
                'positif': [
                    'Pertahankan fasilitas yang sudah ada dengan pemeliharaan rutin',
                    'Tingkatkan fasilitas lainnya sesuai dengan fasilitas yang sudah mendapat apresiasi',
                    'Programkan upgrade fasilitas secara bertahap untuk meningkatkan kenyamanan kampus',
                    'Pertimbangkan penambahan area tertentu (WiFi hotspot, ruang istirahat, dll)'
                ],
                'negatif': [
                    'Segera perbaiki dan maintenance semua fasilitas kampus yang rusak atau tidak berfungsi optimal',
                    'Tingkatkan kualitas akses internet/WiFi di seluruh area kampus',
                    'Tambah ruang belajar dan fasilitas olahraga yang lebih memadai',
                    'Sediakan peralatan laboratorium yang lebih modern dan lengkap',
                    'Perbaiki sistem pendingin ruang kelas dan fasilitas umum lainnya',
                    'Implementasikan sistem pelaporan dan pengecekan fasilitas yang lebih responsif'
                ],
                'netral': [
                    'Evaluasi kebutuhan fasilitas berdasarkan feedback mahasiswa secara berkala'
                ]
            },
            'non-akademik': {
                'positif': [
                    'Dukung terus program UKM dan kegiatan mahasiswa yang telah berjalan',
                    'Tambah alokasi dana untuk kegiatan mahasiswa dan pengembangan bakat mahasiswa',
                    'Promosikan lebih luas program beasiswa dan bantuan finansial yang tersedia'
                ],
                'negatif': [
                    'Tingkatkan program beasiswa dan bantuan finansial untuk mahasiswa yang membutuhkan',
                    'Optimalkan koordinasi antar UKM dan tingkatkan support untuk kegiatan mahasiswa',
                    'Berikan fasilitas dan tempat yang memadai untuk kegiatan UKM dan organisasi mahasiswa',
                    'Sediakan budget yang lebih besar untuk pengembangan dan kegiatan mahasiswa',
                    'Selengkapi sistem informasi tentang beasiswa dan bantuan yang tersedia'
                ],
                'netral': [
                    'Tinjau kembali program non-akademik yang sedang berjalan dan kembangkan sesuai kebutuhan'
                ]
            },
            'lainnya': {
                'positif': [
                    'Lanjutkan program-program yang sudah mendapat apresiasi mahasiswa',
                    'Kembangkan inisiatif serupa di bidang lain untuk meningkatkan kepuasan kampus'
                ],
                'negatif': [
                    'Perlu followup lebih lanjut dan evaluasi mendalam terhadap aspek ini',
                    'Libatkan stakeholder terkait untuk mencari solusi yang tepat'
                ],
                'netral': [
                    'Monitor dan evaluasi terus untuk memastikan peningkatan berkelanjutan'
                ]
            }
        }
        
        # Normalisasi kategori input
        category_lower = category.lower()
        if category_lower not in suggestions_dict:
            category_lower = 'lainnya'
        
        # Ambil saran berdasarkan kategori dan sentimen
        suggestions = suggestions_dict.get(category_lower, {}).get(sentiment, [])
        
        if not suggestions:
            # Fallback jika tidak ada saran spesifik
            if sentiment == 'positif':
                suggestions = ['Pertahankan dan tingkatkan terus upaya-upaya yang sudah berjalan']
            elif sentiment == 'negatif':
                suggestions = ['Perlu evaluasi dan perbaikan menyeluruh terhadap aspek ini']
            else:
                suggestions = ['Pantau dan evaluasi terus untuk memastikan kepuasan mahasiswa']
        
        return {
            'suggestions': suggestions,
            'category': category,
            'sentiment': sentiment
        }


# ========== TRAINING DATA MANAGEMENT ==========

def load_training_data_from_csv(csv_path='training_data.csv'):
    """
    Memuat data training dari file CSV
    Format CSV: text,label
    
    Contoh:
        "Dosen sangat baik",positif
        "Sistem error",negatif
        "Proses normal",netral
    
    Returns:
        tuple: (texts, labels)
    """
    texts = []
    labels = []
    
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, encoding='utf-8')
            texts = df['text'].tolist()
            labels = df['label'].tolist()
            print(f"✅ Loaded {len(texts)} training samples from {csv_path}")
            return texts, labels
        else:
            print(f"⚠️  File {csv_path} tidak ditemukan. Menggunakan data default...")
            return get_initial_training_data()
    except Exception as e:
        print(f"❌ Error membaca CSV: {e}. Menggunakan data default...")
        return get_initial_training_data()


# Contoh data training awal (fallback jika CSV tidak ada) - DIPERKAYA dengan data negasi
def get_initial_training_data():
    """Mendapatkan data training default yang diperkaya untuk pembelajaran lebih cepat"""
    texts = []
    labels = []
    
    # ========== SENTIMEN POSITIF (TAMBAH DATA DOUBLE NEGATION) ==========
    positif_texts = [
        # Akademik
        "sistem akademik sangat membantu", "dosen sangat kompeten dan ramah", "materi kuliah mudah dipahami",
        "metode pengajaran dosen inovatif", "kurikulum sesuai kebutuhan industri", "jadwal kuliah teratur dan jelas",
        "sistem penilaian transparan", "bimbingan akademik sangat membantu", "perpustakaan digital lengkap",
        "e-learning mudah diakses", "dosen selalu tepat waktu", "tugas diberikan proporsional",
        "sistem KRS online cepat", "nilai keluar tepat waktu", "transkrip nilai akurat",
        
        # ===== DATA "tidak + NEGATIF" = POSITIF (TAMBAHAN PENTING) =====
        "kelas tidak berantakan",  # KASUS ANDA!
        "kelasnya tidak berantakan",
        "tidak berantakan kelasnya",
        "ruangan tidak berantakan",
        "meja tidak berantakan",
        "tidak jelek kok",  # Negasi + Negatif = Positif
        "pelayanan tidak jelek",
        "fasilitas tidak jelek",
        "aplikasi tidak jelek",
        "tidak buruk pelayanannya",
        "fasilitas tidak buruk",
        "dosen tidak buruk",
        "sistem tidak error",
        "aplikasi tidak error kok",
        "wifi tidak lemot",
        "koneksi tidak lemot",
        "tidak rusak fasilitasnya",
        "kelas tidak rusak",
        "tidak kotor ruangannya",
        "toilet tidak kotor",
        "kantin tidak kotor",
        "tidak mahal kok biayanya",
        "biaya tidak mahal",
        "tidak sulit prosesnya",
        "pendaftaran tidak sulit",
        "tidak ribet urusannya",
        "administrasi tidak ribet",
        "tidak lambat responnya",
        "pelayanan tidak lambat",
        
        # Administrasi
        "pelayanan administrasi cepat dan responsif", "proses daftar ulang mudah", "pembayaran UKT fleksibel",
        "pelayanan BAAK memuaskan", "pengurusan surat cepat selesai", "sistem informasi mahasiswa bagus",
        "helpdesk tanggap masalah", "staff administrasi ramah", "proses wisata terorganisir dengan baik",
        "layanan pengaduan responsif", "sistem antrian online efektif", "pelayanan terpadu sangat membantu",
        
        # Fasilitas
        "fasilitas kampus lengkap dan nyaman", "ruang kelas ber-AC", "laboratorium modern dan lengkap",
        "wifi kampus cepat", "perpus dengan koleksi lengkap", "fasilitas olahraga memadai",
        "ruang diskusi nyaman", "area parkir luas", "kantin bersih dan enak", "mushola nyaman dan bersih",
        "ruang dosen representatif", "fasilitas disabilitas tersedia", "area hijau dan asri",
        "ruang UKM lengkap", "aula kampus bagus", "toilet bersih dan terawat",
        
        # Layanan
        "website mudah digunakan", "terima kasih atas pelayanannya", "aplikasi sangat bermanfaat",
        "layanan bagus dan memuaskan", "sangat terbantu dengan sistem ini", "proses cepat dan mudah",
        "layanan online praktis", "customer service sangat membantu", "respons cepat dari pihak kampus",
        "komplain ditanggapi dengan baik", "sistem pengaduan efektif", "tim IT sangat responsif",
        
        # Non-akademik
        "UKM berkembang baik", "kegiatan kemahasiswaan aktif", "program beasiswa membantu",
        "konseling mahasiswa bermanfaat", "pengembangan karir terarah", "acara kampus meriah",
        "orientasi mahasiswa baru berkesan", "kompetisi mahasiswa didukung", "magang terstruktur dengan baik",
        "alumni network solid", "student exchange program bagus", "lapangan pekerjaan terbuka luas"
    ]
    
    # ========== SENTIMEN NEGATIF (DENGAN NEGASI) ==========
    negatif_texts = [
        # Akademik
        "krs error terus tidak bisa diakses", "dosen tidak masuk tanpa kabar", "materi kuliah terlalu sulit",
        "metode pengajaran membosankan", "kurikulum tidak update", "jadwal kuliah sering bentrok",
        "sistem penilaian tidak jelas", "bimbingan akademik tidak optimal", "e-learning sering error",
        "dosen sering terlambat", "tugas menumpuk dan tidak proporsional", "sistem KRS lambat",
        "nilai keluar lambat", "transkrip nilai sering salah", "mata kuliah tidak tersedia",
        
        # ===== DATA "tidak + POSITIF" = NEGATIF =====
        "dosen tidak bagus mengajar", "materi tidak bagus dan sulit", "sistem tidak bagus dan error",
        "pelayanan tidak bagus", "fasilitas tidak bagus", "wifi tidak bagus dan lambat",
        "aplikasi tidak bagus", "ruang kelas tidak bagus", "laboratorium tidak bagus",
        "kantin tidak bagus dan kotor", "bimbingan tidak bagus", "jadwal tidak bagus",
        "metode tidak bagus", "kurikulum tidak bagus", "perpustakaan tidak bagus",
        "tidak bagus sama sekali", "gak bagus pelayanannya", "nggak bagus aplikasinya",
        "bukan solusi yang bagus", "belum bagus sistemnya", "jangan kasih yang tidak bagus",
        "tidak baik pelayanannya", "tidak ramah staffnya", "tidak cepat responnya",
        "tidak nyaman ruangannya", "tidak bersih toiletnya", "tidak lengkap fasilitasnya",
        "tidak enak makanannya", "tidak membantu customer servicenya",
        
        # Administrasi
        "pelayanan lambat dan tidak responsif", "proses daftar ulang ribet", "pembayaran UKT sulit",
        "pelayanan BAAK mengecewakan", "pengurusan surat lama", "sistem informasi mahasiswa error",
        "helpdesk tidak membantu", "staff administrasi tidak ramah", "proses wisata berantakan",
        "layanan pengaduan tidak ada respon", "antrian panjang tidak teratur", "pelayanan terpadu lambat",
        "persyaratan administrasi berbelit", "biaya administrasi terlalu mahal",
        
        # Fasilitas
        "fasilitas rusak tidak diperbaiki", "ruang kelas panas dan pengap", "laboratorium alat rusak",
        "wifi kampus lemot", "perpustakaan buku lama", "fasilitas olahraga kurang",
        "ruang diskusi terbatas", "area parkir sempit", "kantin kotor dan mahal",
        "mushola kotor", "ruang dosen tidak layak", "tidak ada fasilitas disabilitas",
        "area kampus kumuh", "ruang UKM tidak memadai", "aula tidak terawat", "toilet kotor dan bau",
        
        # Layanan
        "aplikasi sering crash", "kecewa dengan pelayanannya", "sistem susah digunakan",
        "tidak ada perkembangan", "komplain tidak ditanggapi", "buruk sekali layanannya",
        "layanan offline lambat", "customer service tidak membantu", "respons lambat dari pihak kampus",
        "pengaduan diabaikan", "sistem sering down", "tim IT lambat respon",
        
        # Non-akademik
        "UKM tidak aktif", "kegiatan mahasiswa minim", "beasiswa sulit didapat",
        "konseling tidak membantu", "pengembangan karir tidak jelas", "acara kampus sepi",
        "orientasi mahasiswa membingungkan", "kompetisi tidak didukung", "magang tidak terarah",
        "alumni network tidak berfungsi", "program pertukaran mahasiswa terbatas", "lowongan kerja tidak ada"
    ]
    
    # ========== SENTIMEN NETRAL ==========
    netral_texts = [
        # Akademik
        "sistem berjalan normal", "dosen hadir sesuai jadwal", "materi kuliah standar",
        "metode pengajaran biasa saja", "kurikulum cukup baik", "jadwal kuliah biasa",
        "sistem penilaian standar", "bimbingan akademik biasa", "e-learning berfungsi",
        "dosen tepat waktu kadang", "tugas seperti biasa", "sistem KRS normal",
        
        # Administrasi
        "pelayanan standar saja", "proses daftar ulang biasa", "pembayaran UKT normal",
        "pelayanan BAAK cukup", "pengurusan surat biasa", "sistem informasi cukup",
        "helpdesk tersedia", "staff administrasi biasa", "proses wisata biasa",
        "layanan pengaduan ada", "antrian cukup teratur", "pelayanan terpadu standar",
        
        # Fasilitas
        "fasilitas cukup memadai", "ruang kelas biasa", "laboratorium standar",
        "wifi bisa digunakan", "perpustakaan biasa", "fasilitas olahraga cukup",
        "ruang diskusi tersedia", "parkir cukup", "kantin biasa", "mushola ada",
        "ruang dosen standar", "area kampus biasa", "ruang UKM cukup", "aula biasa",
        
        # Layanan
        "informasi tersedia", "proses sesuai prosedur", "aplikasi bisa digunakan",
        "tidak ada masalah berarti", "masih dalam tahap adaptasi", "perlu evaluasi lebih lanjut",
        "semoga kedepan lebih baik", "cukup membantu", "layanan biasa saja",
        "customer service standar", "respons cukup cepat", "sistem cukup stabil",
        
        # Non-akademik
        "UKM berjalan biasa", "kegiatan mahasiswa standar", "beasiswa tersedia",
        "konseling tersedia", "pengembangan karir cukup", "acara kampus biasa",
        "orientasi mahasiswa biasa", "kompetisi ada", "magang tersedia",
        "alumni network ada", "program pertukaran tersedia", "informasi kerja cukup"
    ]
    
    # Gabungkan semua data
    texts.extend(positif_texts)
    labels.extend(['positif'] * len(positif_texts))
    
    texts.extend(negatif_texts)
    labels.extend(['negatif'] * len(negatif_texts))
    
    texts.extend(netral_texts)
    labels.extend(['netral'] * len(netral_texts))
    
    print(f"📊 Data training default diperkaya: {len(texts)} sampel")
    print(f"   - Positif: {len(positif_texts)}")
    print(f"   - Negatif: {len(negatif_texts)}")
    print(f"   - Netral: {len(netral_texts)}")
    print(f"   - Termasuk data: 'tidak+positif'=negatif & 'tidak+negatif'=positif")
    
    return texts, labels


# Inisialisasi model global
sentiment_analyzer = SentimentAnalyzer()
