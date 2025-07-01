from flask import Flask, render_template
from flask_cors import CORS
from config import Config
from models import db, bcrypt
from routes import auth

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize extensions
    db.init_app(app)
    bcrypt.init_app(app)
    
    # Enable CORS (restrict origins for security)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register blueprints
    app.register_blueprint(auth, url_prefix='/api')

    @app.route('/')
    def home():
        return render_template('login.html')

    return app

if __name__ == '__main__':
    app = create_app()
    with app.app_context():
        try:
            db.create_all()
            print("Database tables created successfully.")
        except Exception as e:
            print(f"Error creating database tables: {e}")
            
    app.run(debug=True)
