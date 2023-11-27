from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pymysql
import threading
import time
from twilio.rest import Client
import configparser
from flask_cors import CORS



app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://admin:capstone6@capstonedatabase.cev18xadhymx.ap-northeast-2.rds.amazonaws.com:3306/capstone'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 모델 정의
class Member(db.Model):
    __tablename__ = 'member'
    memberID = db.Column(db.String(40), primary_key=True)
    memberPW = db.Column(db.String(100), nullable=False)
    memberAD = db.Column(db.String(45), nullable=False)
    memberphone = db.Column(db.String(20), nullable=False)
    cameraID = db.Column(db.Integer, nullable=False)
    alertphonenumber1 = db.Column(db.String(30), nullable=True)
    alertphonenumber2 = db.Column(db.String(30), nullable=True)
    alertphonenumber3 = db.Column(db.String(30), nullable=True)

# 낙상 상태를 저장하는 변수
fall_status = False

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    hashed_password = generate_password_hash(data['memberPW'], method='sha256')
    new_member = Member(
        memberID=data['memberID'],
        memberPW=hashed_password,
        memberAD=data['memberAD'],
        memberphone=data['memberphone'],
        cameraID=1
    )
    db.session.add(new_member)
    db.session.commit()
    return jsonify({'message': 'Member created'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    member = Member.query.filter_by(memberID=data['memberID']).first()
    if member and check_password_hash(member.memberPW, data['memberPW']):
        return jsonify({'message': 'Login successful'}), 200
    else:
        return jsonify({'message': 'Invalid username or password'}), 401

@app.route('/update', methods=['POST'])
def update_member():
    data = request.json
    member = Member.query.get(data['memberID'])
    if member:
        member.memberAD = data.get('memberAD', member.memberAD)
        member.alertphonenumber1 = data.get('alertphonenumber1', member.alertphonenumber1)
        member.alertphonenumber2 = data.get('alertphonenumber2', member.alertphonenumber2)
        member.alertphonenumber3 = data.get('alertphonenumber3', member.alertphonenumber3)
        db.session.commit()
        print("Member updated:", data['memberID'])
        return jsonify({'message': 'Member updated'}), 200
    else:
        print("Member not found:", data['memberID'])
        return jsonify({'message': 'Member not found'}), 404

@app.route('/fall_status', methods=['POST'])
def update_fall_status():
    global fall_status
    data = request.json
    if 'fall_status' in data:
        fall_status = data['fall_status']
        print("Fall status updated:", fall_status)
        return jsonify({'message': 'Fall status updated'}), 200
    else:
        print("Invalid data")
        return jsonify({'message': 'Invalid data'}), 400

@app.route('/get_fall_status', methods=['GET'])
def get_fall_status():
    global fall_status
    return jsonify({'fall_status': fall_status}), 200


@app.route('/checkUserId', methods=['GET'])
def check_user_id():
    user_id = request.args.get('userId')
    if not user_id:
        print("Invalid request")
        return jsonify({'message': 'Invalid request', 'available': False}), 400

    user = Member.query.filter_by(memberID=user_id).first()
    if user:
        return jsonify({'available': False}), 200
    else:
        return jsonify({'available': True}), 200

@app.route('/update_address', methods=['POST'])
def update_address():
    data = request.json
    user_id = data.get('user_id')
    new_address = data.get('new_address')

    if not user_id or not new_address:
        print("Invalid data")
        return jsonify({'message': 'Invalid data'}), 400

    user = Member.query.filter_by(memberID=user_id).first()
    if user:
        user.memberAD = new_address
        db.session.commit()
        print("Address updated:", user_id)
        return jsonify({'message': 'Address updated'}), 200
    else:
        print("User not found:", user_id)
        return jsonify({'message': 'User not found'}), 404

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'ok'}), 200

def check_database_connection():
    try:
        with app.app_context():
            # 간단한 쿼리를 실행하여 데이터베이스 연결 상태를 확인합니다.
            exists = db.session.query(db.exists().where(Member.memberID == 'any_value')).scalar()
            print("Connected to the database successfully.")
    except Exception as e:
        print(f"Failed to connect to the database. Error: {e}")

if __name__ == '__main__':
    check_database_connection()
    app.run(host='0.0.0.0', port=5000)