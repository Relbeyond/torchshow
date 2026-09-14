# 定义类
class Person:
    # 构造方法:创建实例时自动调用，初始化对象的属性
    def __init__(self, name, age):
        # self代表实例对象本身，必须写在方法第一个参数
        self.name = name
        self.age = age

    def say_hello(self):
        print(f"我是{self.name},今年{self.age}岁") # f‑string格式化字符串

p1 = Person("张三", 18)
p1.say_hello()
print(p1.name, p1.age)