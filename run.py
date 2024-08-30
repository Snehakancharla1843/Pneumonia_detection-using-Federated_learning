from app import app, db

if __name__ == '__main__':
    db.create_all()  # Create database tables if they don't exist
    app.run(debug=True)  # Run the Flask app
