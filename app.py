from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import hashlib
import os
import joblib
import numpy as np
from datetime import datetime
 
app = Flask(__name__)
app.secret_key = 'loan_prediction_secret_key_2024'
 
# ─── Load ML Model ─────────────────────────────────────────────────────────────
model = joblib.load('ml_model/loan_model.pkl')
encoders = joblib.load('ml_model/encoders.pkl')
features = joblib.load('ml_model/features.pkl')
 
# ─── Database Setup ─────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect('loan_system.db')
    conn.row_factory = sqlite3.Row
    return conn
 
def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        gender TEXT,
        married TEXT,
        dependents REAL,
        education TEXT,
        self_employed TEXT,
        applicant_income REAL,
        coapplicant_income REAL,
        loan_amount REAL,
        loan_term REAL,
        credit_history REAL,
        property_area TEXT,
        result TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    conn.commit()
    conn.close()
 
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
 
# ─── Home ───────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('home.html')
 
# ─── Signup ─────────────────────────────────────────────────────────────────────
@app.route('/signup/', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = hash_password(request.form['password'])
        conn = get_db()
        try:
            conn.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                         (name, email, password))
            conn.commit()
            flash('Account created! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already registered.', 'danger')
        finally:
            conn.close()
    return render_template('signup.html')
 
# ─── Login ──────────────────────────────────────────────────────────────────────
@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = hash_password(request.form['password'])
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE email=? AND password=?',
                            (email, password)).fetchone()
        conn.close()
        if user:
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            return redirect(url_for('dashboard'))
        flash('Invalid credentials.', 'danger')
    return render_template('login.html')
 
# ─── Logout ─────────────────────────────────────────────────────────────────────
@app.route('/logout/')
def logout():
    session.clear()
    return redirect(url_for('home'))
 
# ─── Change Password ─────────────────────────────────────────────────────────────
@app.route('/change-password/', methods=['GET', 'POST'])
def change_password():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        old_pw = hash_password(request.form['old_password'])
        new_pw = hash_password(request.form['new_password'])
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE id=? AND password=?',
                            (session['user_id'], old_pw)).fetchone()
        if user:
            conn.execute('UPDATE users SET password=? WHERE id=?', (new_pw, session['user_id']))
            conn.commit()
            flash('Password changed successfully!', 'success')
        else:
            flash('Old password is incorrect.', 'danger')
        conn.close()
    return render_template('change_password.html')
 
# ─── Dashboard ──────────────────────────────────────────────────────────────────
@app.route('/dashboard/')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = get_db()
    recent = conn.execute(
        'SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 5',
        (session['user_id'],)).fetchall()
    conn.close()
    return render_template('dashboard.html', recent=recent)
 
# ─── Predict ────────────────────────────────────────────────────────────────────
@app.route('/predict/', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
 
    gender = request.form.get('gender', 'Male')
    married = request.form.get('married', 'No')
    dependents = float(request.form.get('dependents', 0))
    education = request.form.get('education', 'Graduate')
    self_employed = request.form.get('self_employed', 'No')
    applicant_income = float(request.form.get('applicant_income', 0))
    coapplicant_income = float(request.form.get('coapplicant_income', 0))
    loan_amount = float(request.form.get('loan_amount', 0))
    loan_term = float(request.form.get('loan_term', 360))
    credit_history = float(request.form.get('credit_history', 1))
    property_area = request.form.get('property_area', 'Urban')
 
    # Encode inputs
    gender_enc = encoders['Gender'].transform([gender])[0]
    married_enc = encoders['Married'].transform([married])[0]
    edu_enc = encoders['Education'].transform([education])[0]
    self_enc = encoders['Self_Employed'].transform([self_employed])[0]
    prop_enc = encoders['Property_Area'].transform([property_area])[0]
 
    input_data = np.array([[gender_enc, married_enc, dependents, edu_enc, self_enc,
                            applicant_income, coapplicant_income, loan_amount,
                            loan_term, credit_history, prop_enc]])
 
    prediction = model.predict(input_data)[0]
    result = 'Approved' if prediction == 1 else 'Rejected'
 
    # Save to DB
    conn = get_db()
    conn.execute('''INSERT INTO predictions
        (user_id, gender, married, dependents, education, self_employed,
         applicant_income, coapplicant_income, loan_amount, loan_term,
         credit_history, property_area, result)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (session['user_id'], gender, married, dependents, education, self_employed,
         applicant_income, coapplicant_income, loan_amount, loan_term,
         credit_history, property_area, result))
    conn.commit()
    conn.close()
 
    return render_template('result.html', result=result,
                           income=applicant_income, loan=loan_amount)
 
# ─── History ────────────────────────────────────────────────────────────────────
@app.route('/history/')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = get_db()
    all_preds = conn.execute(
        'SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC',
        (session['user_id'],)).fetchall()
    conn.close()
    return render_template('history.html', predictions=all_preds)
 
# ─── Admin Login ─────────────────────────────────────────────────────────────────
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin123'
 
@app.route('/admin-login/', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
        flash('Invalid admin credentials.', 'danger')
    return render_template('admin_login.html')
 
@app.route('/admin-logout/')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('home'))
 
# ─── Admin Dashboard ─────────────────────────────────────────────────────────────
@app.route('/admin-dashboard/')
def admin_dashboard():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    conn = get_db()
    total_users = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    total_preds = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    approved = conn.execute("SELECT COUNT(*) FROM predictions WHERE result='Approved'").fetchone()[0]
    rejected = conn.execute("SELECT COUNT(*) FROM predictions WHERE result='Rejected'").fetchone()[0]
    conn.close()
    return render_template('admin_dashboard.html',
                           total_users=total_users, total_preds=total_preds,
                           approved=approved, rejected=rejected)
 
# ─── Admin Manage Users ───────────────────────────────────────────────────────────
@app.route('/admin-users/')
def admin_users():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    conn = get_db()
    users = conn.execute('SELECT * FROM users ORDER BY created_at DESC').fetchall()
    conn.close()
    return render_template('admin_users.html', users=users)
 
@app.route('/admin-delete-user/<int:user_id>')
def admin_delete_user(user_id):
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    conn = get_db()
    conn.execute('DELETE FROM predictions WHERE user_id=?', (user_id,))
    conn.execute('DELETE FROM users WHERE id=?', (user_id,))
    conn.commit()
    conn.close()
    flash('User deleted.', 'success')
    return redirect(url_for('admin_users'))
 
# ─── Admin View Predictions ───────────────────────────────────────────────────────
@app.route('/admin-predictions/')
def admin_predictions():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    conn = get_db()
    preds = conn.execute('''SELECT p.*, u.name, u.email FROM predictions p
                            JOIN users u ON p.user_id = u.id
                            ORDER BY p.created_at DESC''').fetchall()
    conn.close()
    return render_template('admin_predictions.html', predictions=preds)
 
if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=8000)