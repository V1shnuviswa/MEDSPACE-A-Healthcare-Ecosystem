from flask import Blueprint, request, jsonify
from models import db, User
import traceback  # Import this for error tracking

auth = Blueprint('auth', __name__)

@auth.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        if not data.get('email') or not data.get('password') or not data.get('user_type'):
            return jsonify({'error': 'Missing required fields'}), 400

        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400

        user = User(
            full_name=data.get('full_name', ''),
            email=data['email'],
            user_type=data['user_type']
        )
        user.set_password(data['password'])
        db.session.add(user)
        db.session.commit()

        return jsonify({'message': 'User registered successfully'}), 201

    except Exception as e:
        print("Error in Register:", str(e))
        traceback.print_exc()
        return jsonify({'error': 'Internal Server Error'}), 500


@auth.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        user = User.query.filter_by(email=data['email']).first()

        if user and user.check_password(data['password']):
            return jsonify({'message': 'Login successful'}), 200
        
        return jsonify({'error': 'Invalid credentials'}), 401

    except Exception as e:
        print("Error in Login:", str(e))
        traceback.print_exc()
        return jsonify({'error': 'Internal Server Error'}), 500
