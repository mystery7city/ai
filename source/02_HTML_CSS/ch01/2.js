// 2.js:동적인 부분(파일 -> 다른 이름으로 저장 - > UTF8 선택 후 저장)
name = prompt("이름은?", "홍길동") // 취소를 클릭시 'null' 리턴
if (nmae != 'null') {
    document.write(name + '님 반갑습니다.<br>');
}