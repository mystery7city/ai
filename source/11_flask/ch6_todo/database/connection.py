# pip install python-dotenv
# pip freeze > requirements.txt
import oracledb
import os
from dotenv import load_dotenv
# conn = cx_Oracle.connect("scott", "tiger", "127.0.0.1:1521/xe")
load_dotenv()
dbserver_ip = os.getenv('DBSERVER_IP')
oracle_port = os.getenv('ORACLE_PORT')
oracle_user = os.getenv('ORACLE_USER')
oracle_password = os.getenv('ORACLE_PASSWORD')
conn = oracledb.connect(oracle_user, 
                oracle_password,
                f"{dbserver_ip}:{oracle_port}/xe")