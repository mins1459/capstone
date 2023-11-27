import configparser
from cryptography.fernet import Fernet

# 암호화 함수
def encrypt_message(message):
    key = open("secret.key", "rb").read()
    f = Fernet(key)
    encrypted_message = f.encrypt(message.encode())
    return encrypted_message.decode()

# config.ini 파일 읽기
config = configparser.ConfigParser()
config.read('config.ini')

# TWILIO 섹션 암호화
config.set('TWILIO', 'ACCOUNT_SID', encrypt_message(config.get('TWILIO', 'ACCOUNT_SID')))
config.set('TWILIO', 'AUTH_TOKEN', encrypt_message(config.get('TWILIO', 'AUTH_TOKEN')))
config.set('TWILIO', 'PHONE_NUMBER', encrypt_message(config.get('TWILIO', 'PHONE_NUMBER')))

# DATABASE 섹션 암호화
config.set('DATABASE', 'HOST', encrypt_message(config.get('DATABASE', 'HOST')))
config.set('DATABASE', 'PORT', encrypt_message(str(config.get('DATABASE', 'PORT'))))  # PORT는 정수이므로 문자열로 변환
config.set('DATABASE', 'USER', encrypt_message(config.get('DATABASE', 'USER')))
config.set('DATABASE', 'PASSWORD', encrypt_message(config.get('DATABASE', 'PASSWORD')))
config.set('DATABASE', 'DB_NAME', encrypt_message(config.get('DATABASE', 'DB_NAME')))
config.set('DATABASE', 'CHARSET', encrypt_message(config.get('DATABASE', 'CHARSET')))

# 암호화된 내용을 config.ini 파일에 다시 쓰기
with open('config.ini', 'w') as configfile:
    config.write(configfile)
