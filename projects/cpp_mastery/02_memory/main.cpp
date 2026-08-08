#include <iostream>

int main()
{
    int number{10};
    int* pointer{&number};

    std::cout << "number value: " << number << '\n';
    std::cout << "number address: " << &number << '\n';

    std::cout << "pointer value: " << pointer << '\n';
    std::cout << "pointed value: " << *pointer << '\n';

    *pointer = 25;

    std::cout << "changed number: " << number << '\n';

    return 0;
}