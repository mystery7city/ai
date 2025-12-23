# PowerShell 관리자 모드
# app.py 생성 후 ctrl+j 터미널 창을 열기 
# 가상환경 만들기 : python -m
# citrl + shift + p -> 인터프리터 선택 -> .venv 가상환경 선택
from flask import Flask
app = Flask(__name__) # 웹서버 객체(앱 인스턴스 생성)

@app.route("/") # 데코레이터를 통해 가능한 url 등록
def main_handler():
    return "<H1>Hello, World</H1>"
@app.route("/apt")
def apt_handler():
    # return "<h1>예상 금액은 1,000원입니다</h1>"
    return{
        'price':'1,000',
        'unit':'won'
    }
# 실행 : flask run -- port=80 -- debug
# app.py가 아닌 파일 프랄크스 실행 : python ex1_app.py
if __name__=="__main__":
    app.run(port=80, debug=True) # 소스 수정시 서버 자동 재시작, port=80번