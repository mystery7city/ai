# 데코레이터(@) : 대상함수를 

def check(func):
    def wrapper():
        print(func.__name__, '함수 전처리')
        func
        print(func.__name__, '함수 전처리')
    return wrapper

def hello():
    print(hello.__name__, '함수 전터리')
    print('Hello')
    print(hello.__name__, '함수 전터리')

def world():
    print(world.__name__, '함수 전처리')
    print('world')
    print(world.__name__, '함수 후처리')

if __name__=="__main__":
    wrapper_hello = check(hello)
    wrapper_hello()
    wrapper_world = check(hello)
