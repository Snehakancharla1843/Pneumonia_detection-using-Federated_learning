import os

class Config:
    SECRET_KEY = os.urandom(24).hex()
    SQLALCHEMY_DATABASE_URI = 'sqlite:///site.db'
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
