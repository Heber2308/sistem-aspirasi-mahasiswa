#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk setup database MySQL
Membantu membuat database dan import SQL schema
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_mysql():
    """
    Setup MySQL database dengan import aspirasi_mysql.sql
    """
    print("=" * 60)
    print("🔧 SETUP DATABASE MYSQL - WEBSITE ASPIRASI MAHASISWA")
    print("=" * 60)
    
    # Input konfigurasi MySQL
    print("\n📝 Masukkan konfigurasi MySQL Anda:")
    
    db_host = input("Host MySQL [localhost]: ").strip() or "localhost"
    db_port = input("Port MySQL [3306]: ").strip() or "3306"
    db_user = input("Username MySQL [root]: ").strip() or "root"
    db_password = input("Password MySQL [kosongkan jika tidak ada]: ").strip()
    db_name = input("Nama database [aspirasi_mahasiswa]: ").strip() or "aspirasi_mahasiswa"
    
    print("\n" + "=" * 60)
    print("📊 Konfigurasi MySQL Anda:")
    print("=" * 60)
    print(f"  Host: {db_host}")
    print(f"  Port: {db_port}")
    print(f"  Username: {db_user}")
    print(f"  Password: {'*' * len(db_password) if db_password else '(kosong)'}")
    print(f"  Database: {db_name}")
    print("=" * 60)
    
    # Simpan ke .env file
    env_file = Path(".env")
    env_content = f"""# MySQL Configuration
DB_HOST={db_host}
DB_PORT={db_port}
DB_USER={db_user}
DB_PASSWORD={db_password}
DB_NAME={db_name}
FLASK_ENV=development
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    print(f"\n✅ File .env berhasil dibuat")
    
    # Set environment variables
    os.environ['DB_HOST'] = db_host
    os.environ['DB_PORT'] = db_port
    os.environ['DB_USER'] = db_user
    os.environ['DB_PASSWORD'] = db_password
    os.environ['DB_NAME'] = db_name
    
    print("\n🔄 Membuat database dan import schema...")
    
    # Cek apakah aspirasi_mysql.sql ada
    sql_file = Path("instance/aspirasi_mysql.sql")
    if not sql_file.exists():
        print(f"❌ File tidak ditemukan: {sql_file}")
        print("   Pastikan file instance/aspirasi_mysql.sql ada di folder project")
        return False
    
    try:
        # Import SQL ke MySQL menggunakan mysql command
        if db_password:
            cmd = f'mysql -h {db_host} -u {db_user} -p{db_password} < instance/aspirasi_mysql.sql'
        else:
            cmd = f'mysql -h {db_host} -u {db_user} < instance/aspirasi_mysql.sql'
        
        print(f"\n📌 Menjalankan command:")
        print(f"   {cmd.replace(db_password, '****') if db_password else cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"\n❌ Error saat import SQL:")
            print(result.stderr)
            
            # Coba alternatif dengan Python
            print("\n🔄 Mencoba alternatif dengan PyMySQL...")
            try:
                import pymysql
                print("✅ PyMySQL terinstall, lanjut ke step berikutnya...")
            except ImportError:
                print("⚠️  PyMySQL belum terinstall")
                print("   Install dengan: pip install PyMySQL")
                return False
            
            # Buat database terlebih dahulu jika belum ada
            try:
                import pymysql
                connection = pymysql.connect(
                    host=db_host,
                    user=db_user,
                    password=db_password or None,
                    port=int(db_port)
                )
                cursor = connection.cursor()
                
                # Create database
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
                print(f"✅ Database '{db_name}' siap digunakan")
                
                # Switch ke database
                cursor.execute(f"USE `{db_name}`")
                
                # Read dan execute SQL file
                with open(sql_file, 'r', encoding='utf-8') as f:
                    sql_content = f.read()
                
                # Execute SQL statements
                for statement in sql_content.split(';'):
                    statement = statement.strip()
                    if statement and not statement.startswith('--'):
                        cursor.execute(statement)
                
                connection.commit()
                cursor.close()
                connection.close()
                print(f"✅ Schema berhasil di-import ke database '{db_name}'")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                return False
        else:
            print(f"✅ Schema berhasil di-import")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ SETUP SELESAI!")
    print("=" * 60)
    print(f"""
Database MySQL sudah siap digunakan!

🚀 Untuk menjalankan aplikasi:
   1. Buka terminal di folder project
   2. Jalankan: python app.py
   3. Aplikasi akan berjalan di http://localhost:5000

📝 Login Admin:
   - NIM: ADMIN001
   - Password: admin123

⚙️  Konfigurasi MySQL disimpan di file .env
   Anda bisa mengeditnya jika ada perubahan konfigurasi
    """)
    return True


if __name__ == '__main__':
    try:
        setup_mysql()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup dibatalkan oleh user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error tidak terduga: {e}")
        sys.exit(1)
