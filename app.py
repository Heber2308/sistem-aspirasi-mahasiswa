from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from flask_bcrypt import Bcrypt
import os
from datetime import datetime
import webbrowser
import threading
import time
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables dari .env file
load_dotenv()

# Import modul kita
from database import db, User, Category, Aspiration, SentimentLog
from sentiment_model import sentiment_analyzer, load_training_data_from_csv

# Inisialisasi aplikasi
app = Flask(__name__)

# Configuration - Support development & production
# Configuration - Support development & production
environment = os.environ.get('FLASK_ENV', 'development')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'rahasia-kampus-2026')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ========== DATABASE CONFIGURATION ==========
# ========== DATABASE CONFIGURATION ==========
# Cek apakah ada environment variable dari Railway (MYSQLHOST, dll)
if os.environ.get('MYSQLHOST'):
    # Railway memberikan MYSQLHOST, MYSQLPORT, MYSQLUSER, MYSQLPASSWORD, MYSQLDATABASE
    mysql_user = os.environ.get('MYSQLUSER')
    mysql_pass = os.environ.get('MYSQLPASSWORD')
    mysql_host = os.environ.get('MYSQLHOST')
    mysql_port = os.environ.get('MYSQLPORT')
    mysql_db = os.environ.get('MYSQLDATABASE')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{mysql_user}:{mysql_pass}@{mysql_host}:{mysql_port}/{mysql_db}'
    print("✅ Menggunakan koneksi database dari Railway")
else:
    # Menggunakan variable sendiri (DB_HOST, DB_NAME, dll) atau default
    mysql_user = os.environ.get('DB_USER', 'root')
    mysql_pass = os.environ.get('DB_PASSWORD', '')
    mysql_host = os.environ.get('DB_HOST', 'localhost')
    mysql_port = os.environ.get('DB_PORT', '3306')
    mysql_db = os.environ.get('DB_NAME', 'aspirasi_mahasiswa')
    
    # Jika ada password, format: user:password@host:port/db
    if mysql_pass:
        app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{mysql_user}:{mysql_pass}@{mysql_host}:{mysql_port}/{mysql_db}'
    else:
        app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{mysql_user}@{mysql_host}:{mysql_port}/{mysql_db}'
    print("✅ Menggunakan koneksi database lokal / custom")
# Inisialisasi ekstensi
CORS(app)
bcrypt = Bcrypt(app)
db.init_app(app)

# ========== INISIALISASI DATABASE ==========
def init_database():
    """Membuat tabel dan data awal"""
    with app.app_context():
        # Debug: Tampilkan database configuration
        db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', 'unknown')
        # Hide password in URI
        masked_uri = db_uri.replace(os.environ.get('DB_PASSWORD', ''), '****') if os.environ.get('DB_PASSWORD') else db_uri
        print(f"\n📊 Database Configuration:")
        print(f"   URI: {masked_uri}")
        print(f"   Host: {os.environ.get('DB_HOST', 'localhost')}")
        print(f"   Database: {os.environ.get('DB_NAME', 'aspirasi_mahasiswa')}")
        
        try:
            db.create_all()
            print("✅ Tabel database berhasil disiapkan")
        except Exception as e:
            print(f"❌ Error saat membuat tabel: {e}")
            print("   Pastikan MySQL server sudah berjalan dan database sudah dibuat")
            print(f"   Jalankan: python setup_mysql.py")
            return False
        
        # Cek apakah ada kategori
        if Category.query.count() == 0:
            categories = [
                Category(name="Akademik", description="Terkait perkuliahan, kurikulum, dosen, jadwal"),
                Category(name="Administrasi", description="Terkait KRS, KHS, UKT, administrasi akademik"),
                Category(name="Fasilitas", description="Terkait fasilitas kampus, laboratorium, perpustakaan"),
                Category(name="Non-Akademik", description="Terkait UKM, beasiswa, kegiatan mahasiswa"),
                Category(name="Lainnya", description="Aspirasi di luar kategori di atas")
            ]
            for cat in categories:
                db.session.add(cat)
            db.session.commit()
            print("✅ Kategori berhasil ditambahkan")
        
        # Cek apakah ada user admin
        if User.query.filter_by(role='admin').count() == 0:
            admin = User(
                nim='ADMIN001',
                nama='Administrator',
                email='admin@kampus.ac.id',
                password=bcrypt.generate_password_hash('admin123').decode('utf-8'),
                fakultas='Fakultas Teknologi Informasi',
                prodi='Sistem Informasi',
                role='admin'
            )
            db.session.add(admin)
            db.session.commit()
            print("✅ User admin berhasil dibuat (NIM: ADMIN001, Password: admin123)")
        
        # Training model AI dengan data awal
        print("\n🤖 Melatih model AI sentimen...")
        texts, labels = load_training_data_from_csv()
        sentiment_analyzer.train(texts, labels)
        print("✅ Model AI siap digunakan")


# ========== ROUTE WEBSITE ==========
@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html')

@app.route('/test')
def test():
    """Route testing - halaman paling sederhana untuk diagnosa"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TEST</title>
        <style>
            body { font-family: Arial; background: #f0f0f0; padding: 50px; text-align: center; }
            h1 { color: #333; }
            .success { background: #4CAF50; color: white; padding: 20px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="success">
            <h1>✅ FLASK BEKERJA NORMAL</h1>
            <p style="font-size: 18px;">Jika anda melihat halaman ini, Flask dan server berfungsi dengan baik.</p>
            <p>Jika halaman index kosong, berarti ada masalah di:</p>
            <ul style="text-align: left; display: inline-block;">
                <li>File <strong>templates/index.html</strong></li>
                <li>Render template tidak load</li>
                <li>CSS/JS dari CDN tidak berhasil</li>
            </ul>
            <h3><a href="/">← Kembali ke halaman utama</a></h3>
        </div>
    </body>
    </html>
    """

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Halaman login"""
    if request.method == 'POST':
        nim = request.form.get('nim')
        password = request.form.get('password')
        
        user = User.query.filter_by(nim=nim).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_nim'] = user.nim
            session['user_nama'] = user.nama
            session['user_role'] = user.role
            
            if user.role == 'admin':
                return redirect(url_for('dashboard_admin'))
            else:
                return redirect(url_for('dashboard_mahasiswa'))
        else:
            return render_template('login.html', error='NIM atau password salah')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Halaman registrasi"""
    if request.method == 'POST':
        nim = request.form.get('nim')
        nama = request.form.get('nama')
        email = request.form.get('email')
        password = request.form.get('password')
        fakultas = request.form.get('fakultas')
        prodi = request.form.get('prodi')
        
        # Cek apakah NIM sudah terdaftar
        existing_user = User.query.filter_by(nim=nim).first()
        if existing_user:
            return render_template('register.html', error='NIM sudah terdaftar')
        
        # Buat user baru
        new_user = User(
            nim=nim,
            nama=nama,
            email=email,
            password=bcrypt.generate_password_hash(password).decode('utf-8'),
            fakultas=fakultas,
            prodi=prodi,
            role='mahasiswa'
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout pengguna"""
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard/mahasiswa')
def dashboard_mahasiswa():
    """Dashboard mahasiswa"""
    if 'user_id' not in session or session.get('user_role') != 'mahasiswa':
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    aspirations = Aspiration.query.filter_by(user_id=user.id).order_by(Aspiration.created_at.desc()).all()
    categories = Category.query.all()
    
    return render_template('dashboard_mahasiswa.html', user=user, aspirations=aspirations, categories=categories)

@app.route('/dashboard/admin')
def dashboard_admin():
    """Dashboard admin"""
    if 'user_id' not in session or session.get('user_role') != 'admin':
        return redirect(url_for('login'))
    
    # Data statistik
    total_aspirations = Aspiration.query.count()
    total_users = User.query.filter_by(role='mahasiswa').count()
    
    aspirations = Aspiration.query.order_by(Aspiration.created_at.desc()).all()
    
    # Statistik sentimen
    sentiment_stats = {
        'positif': Aspiration.query.filter_by(sentiment='positif').count(),
        'netral': Aspiration.query.filter_by(sentiment='netral').count(),
        'negatif': Aspiration.query.filter_by(sentiment='negatif').count()
    }
    
    # Statistik per kategori
    categories = Category.query.all()
    category_stats = []
    for cat in categories:
        cat_stats = {
            'name': cat.name,
            'total': Aspiration.query.filter_by(category_id=cat.id).count(),
            'positif': Aspiration.query.filter_by(category_id=cat.id, sentiment='positif').count(),
            'negatif': Aspiration.query.filter_by(category_id=cat.id, sentiment='negatif').count()
        }
        category_stats.append(cat_stats)
    
    return render_template('dashboard_admin.html', 
                          user=session,
                          total_aspirations=total_aspirations,
                          total_users=total_users,
                          aspirations=aspirations,
                          sentiment_stats=sentiment_stats,
                          category_stats=category_stats)


# ========== API ENDPOINTS ==========
@app.route('/api/aspirasi', methods=['POST'])
def create_aspiration():
    """API untuk membuat aspirasi baru"""
    try:
        data = request.get_json()
        
        # Validasi data
        if not data.get('content'):
            return jsonify({'error': 'Isi aspirasi tidak boleh kosong'}), 400
        
        # Ambil kategori untuk generate saran
        category = Category.query.get(data.get('category_id'))
        category_name = category.name if category else 'lainnya'
        
        # Analisis sentimen
        sentiment_result = sentiment_analyzer.predict(data['content'])
        
        # Generate suggestions
        suggestions_result = sentiment_analyzer.generate_suggestions(
            data['content'],
            sentiment_result['sentiment'],
            category_name
        )
        
        # Konversi suggestions list ke JSON string
        import json
        suggestions_json = json.dumps(suggestions_result['suggestions'])
        
        # Simpan aspirasi
        aspiration = Aspiration(
            user_id=data['user_id'],
            category_id=data['category_id'],
            content=data['content'],
            sentiment=sentiment_result['sentiment'],
            sentiment_score=sentiment_result['confidence'] / 100,
            confidence=sentiment_result['confidence'],
            suggestions=suggestions_json,
            status='pending'
        )
        
        db.session.add(aspiration)
        db.session.commit()
        
        # Simpan log sentimen
        log = SentimentLog(
            aspiration_id=aspiration.id,
            sentiment=sentiment_result['sentiment'],
            confidence=sentiment_result['confidence']
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'id': aspiration.id,
            'sentiment': sentiment_result['sentiment'],
            'confidence': sentiment_result['confidence'],
            'probabilities': sentiment_result['probabilities'],
            'suggestions': suggestions_result['suggestions']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/aspirasi/<int:id>', methods=['GET'])
def get_aspiration(id):
    """API untuk mendapatkan detail aspirasi"""
    import json
    
    aspiration = Aspiration.query.get_or_404(id)
    
    suggestions = []
    if aspiration.suggestions:
        try:
            suggestions = json.loads(aspiration.suggestions)
        except:
            suggestions = []
    
    return jsonify({
        'id': aspiration.id,
        'user_id': aspiration.user_id,
        'user_nama': aspiration.user.nama if aspiration.user else None,
        'category_id': aspiration.category_id,
        'category_name': aspiration.category.name if aspiration.category else None,
        'content': aspiration.content,
        'sentiment': aspiration.sentiment,
        'confidence': aspiration.confidence,
        'suggestions': suggestions,
        'status': aspiration.status,
        'admin_response': aspiration.admin_response,
        'created_at': aspiration.created_at.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/aspirasi/<int:id>/suggestions', methods=['GET'])
def get_suggestions(id):
    """API untuk mendapatkan saran perbaikan dari AI"""
    import json
    
    aspiration = Aspiration.query.get_or_404(id)
    
    suggestions = []
    if aspiration.suggestions:
        try:
            suggestions = json.loads(aspiration.suggestions)
        except:
            suggestions = []
    
    return jsonify({
        'category': aspiration.category.name if aspiration.category else 'Lainnya',
        'sentiment': aspiration.sentiment,
        'suggestions': suggestions
    })

@app.route('/api/aspirasi/<int:id>/response', methods=['PUT'])
def respond_aspiration(id):
    """API untuk memberikan respons admin"""
    if session.get('user_role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    aspiration = Aspiration.query.get_or_404(id)
    
    aspiration.status = data.get('status', aspiration.status)
    aspiration.admin_response = data.get('admin_response')
    aspiration.updated_at = datetime.utcnow()
    
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/stats/sentimen', methods=['GET'])
def get_sentiment_stats():
    """API untuk mendapatkan statistik sentimen dan umum"""
    from sqlalchemy import func
    
    # Statistik Aspirasi
    total = Aspiration.query.count()
    positif = Aspiration.query.filter_by(sentiment='positif').count()
    netral = Aspiration.query.filter_by(sentiment='netral').count()
    negatif = Aspiration.query.filter_by(sentiment='negatif').count()
    
    # Statistik Aspirasi Terselesaikan (status = 'resolved')
    terselesaikan = Aspiration.query.filter_by(status='resolved').count()
    
    # Statistik Mahasiswa Aktif (pengguna yang memiliki setidaknya satu aspirasi)
    mahasiswa_aktif = db.session.query(func.count(func.distinct(Aspiration.user_id))).scalar() or 0
    
    return jsonify({
        'total': total,
        'positif': positif,
        'positif_percent': (positif / total * 100) if total > 0 else 0,
        'netral': netral,
        'netral_percent': (netral / total * 100) if total > 0 else 0,
        'negatif': negatif,
        'negatif_percent': (negatif / total * 100) if total > 0 else 0,
        'mahasiswa_aktif': mahasiswa_aktif,
        'aspirasi_terselesaikan': terselesaikan
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    """API untuk menganalisis sentimen teks"""
    data = request.get_json()
    text = data.get('text', '')
    category = data.get('category', '')
    
    result = sentiment_analyzer.predict(text)
    
    # Generate suggestions
    suggestions = sentiment_analyzer.generate_suggestions(text, result['sentiment'], category)
    result['suggestions'] = suggestions['suggestions']
    
    return jsonify(result)


# ========== FUNGSI PEMBUKA BROWSER (TANPA AUTO-RELOADER) ==========
def open_browser():
    """Buka browser secara otomatis (hanya sekali)"""
    # Dapatkan port dari environment atau default
    port = int(os.environ.get('PORT', 8000))
    
    # Buat lock file di temporary directory
    lock_file = Path(tempfile.gettempdir()) / f'flask_app_port_{port}.lock'
    
    # Cek apakah lock file sudah ada
    if lock_file.exists():
        # Cek umur lock file untuk menghindari stale lock
        lock_age = time.time() - lock_file.stat().st_mtime
        if lock_age < 10:  # Jika kurang dari 10 detik
            print('ℹ️  Browser sudah dibuka sebelumnya, melewati...')
            return
        else:
            # Hapus lock file yang sudah kadaluarsa
            print('🔄 Membersihkan lock file yang kadaluarsa...')
            try:
                lock_file.unlink()
            except:
                pass
    
    # Tunggu sebentar agar server benar-benar siap
    time.sleep(2)
    
    # Buat lock file baru
    try:
        lock_file.touch()
    except:
        pass
    
    # Buka browser
    url = f'http://127.0.0.1:{port}'
    try:
        print(f'\n🌐 Membuka browser ke: {url}')
        webbrowser.open(url)
        print('✅ Browser siap digunakan!')
        print(f'📍 URL: {url}')
        print('📝 Tekan Ctrl+C untuk menghentikan server\n')
    except Exception as e:
        print(f'⚠️  Browser tidak bisa dibuka otomatis')
        print(f'📍 Silakan buka manual: {url}')


# ========== RUN APLIKASI ==========
if __name__ == '__main__':
    # Inisialisasi database
    init_database()
    
    # Jalankan aplikasi
    # Production: debug=False, Development: debug=False (MATIKAN DEBUG MODE)
    is_production = os.environ.get('FLASK_ENV') == 'production'
    port = int(os.environ.get('PORT', 8000))  # Default port 8000
    
    print(f'\n{"="*50}')
    print(f'🚀 SISTEM ASPIRASI MAHASISWA')
    print(f'{"="*50}')
    print(f'📌 Environment: {"PRODUCTION" if is_production else "DEVELOPMENT"}')
    print(f'🔧 Debug Mode: OFF (Tidak ada auto-restart)')
    print(f'🌐 Port: {port}')
    print(f'{"="*50}\n')
    
    # Jika bukan production mode, jalankan browser thread
    if not is_production:
        print('✅ APLIKASI SIAP!')
        print('🌐 Browser akan dibuka secara otomatis...\n')
        
        # Jalankan browser dalam thread terpisah
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
    
    # Jalankan Flask TANPA auto-reloader
    app.run(
        debug=False,  # MATIKAN DEBUG MODE - TIDAK AKAN RESTART OTOMATIS
        host='127.0.0.1',
        port=port,
        use_reloader=False  # MATIKAN AUTO-RELOADER - TIDAK AKAN RESTART SAAT FILE BERUBAH
    )