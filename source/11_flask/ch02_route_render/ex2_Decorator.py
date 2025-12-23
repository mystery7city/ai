# 데코레이터(@) : 대상함수를 

def check(func):
    def wrapper():
        print(func.__name__, '함수 전처리')
        func
        print(func.__name__, '함수 전처리')
    return wrapper

@check
def hello():
    print('Hello')
    
@check
def world():
    print('world')
    

if __name__=="__main__":
    hello()
    world()