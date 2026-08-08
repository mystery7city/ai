#include <iostream>
#include <string>

int main() {
    std::cout << "C++ 학습 프로그램을 시작합니다!" << std::endl;

    std::cout << "이름을 입력하세요: ";
    std::string name;
    std::getline(std::cin, name);

    std::cout << name << "님, C++의 세계에 오신 것을 환영합니다!" << std::endl;

    return 0;
}
