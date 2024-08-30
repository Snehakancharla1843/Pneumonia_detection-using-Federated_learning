from flask import render_template, url_for, flash, redirect, request
from app import app, db
from app.forms import RegistrationForm, LoginForm
from app.models import User
from flask_login import login_user, current_user, logout_user, login_required
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model (make sure the model file is in the appropriate directory)
model = load_model(r'C:\Users\sneha\pfl\models\pneumonia_model.h5')

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data, password=form.password.data)
        db.session.add(user)
        db.session.commit()
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.password == form.password.data:
            login_user(user, remember=True)
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/account')
@login_required
def account():
    return render_template('account.html', title='Account')

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            flash('File successfully uploaded')
            return redirect(url_for('result', filename=filename))
    return render_template('upload.html', title='Upload')

@app.route('/result')
@login_required
def result():
    filename = request.args.get('filename')
    if not filename:
        flash('No file uploaded for analysis')
        return redirect(url_for('upload'))

    # Load and preprocess the uploaded image
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make a prediction using the model
    prediction = model.predict(img_array)[0][0]

    # Determine the prediction category based on custom thresholds
    if prediction < 0.5:
        predicted_class = 'Normal'
        confidence = 1 - prediction  # Confidence for Normal
    elif 0.5 <= prediction < 0.75:
        predicted_class = 'Consult Doctor for Early Prevention'
        confidence = prediction  # Confidence for early prevention
    else:
        predicted_class = 'Pneumonia'
        confidence = prediction  # Confidence for Pneumonia

    # Ensure accuracy is within the range 0 to 1, then multiply by 100 to convert to percentage
    accuracy_percentage = confidence * 100

    return render_template(
        'result.html',
        title='Result',
        prediction=predicted_class,
        accuracy=round(accuracy_percentage, 2)  # Round off to 2 decimal places for better readability
    )
