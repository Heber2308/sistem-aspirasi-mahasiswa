from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    """Model untuk pengguna"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    nim = db.Column(db.String(20), unique=True, nullable=False)
    nama = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    fakultas = db.Column(db.String(50))
    prodi = db.Column(db.String(50))
    role = db.Column(db.String(20), default='mahasiswa')  # mahasiswa, admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relasi ke aspirasi
    aspirations = db.relationship('Aspiration', backref='user', lazy=True)


class Category(db.Model):
    """Model untuk kategori aspirasi"""
    __tablename__ = 'categories'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relasi ke aspirasi
    aspirations = db.relationship('Aspiration', backref='category', lazy=True)


class Aspiration(db.Model):
    """Model untuk aspirasi mahasiswa"""
    __tablename__ = 'aspirations'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('categories.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    
    # Hasil analisis sentimen
    sentiment = db.Column(db.String(20))  # positif, netral, negatif
    sentiment_score = db.Column(db.Float)
    confidence = db.Column(db.Float)
    
    # Saran AI berdasarkan sentimen
    suggestions = db.Column(db.Text)  # JSON array dari saran
    
    
    # Status aspirasi
    status = db.Column(db.String(20), default='pending')  # pending, processing, resolved, rejected
    admin_response = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SentimentLog(db.Model):
    """Log hasil analisis sentimen"""
    __tablename__ = 'sentiment_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    aspiration_id = db.Column(db.Integer, db.ForeignKey('aspirations.id'), nullable=False)
    sentiment = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    processed_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relasi ke aspirasi
    aspiration = db.relationship('Aspiration', backref='sentiment_logs')