---
tags:
  - Python
related:
  - "[[深度学习基础知识点]]"
---

## 1. class —— 定义类

`class` 是创建对象的模板。方法就是定义在类里面的函数。

```python
class Dog:
    def bark(self):
        print("汪汪")
```

## 2. 调用类里面的方法

类里的函数叫"方法"，调用方式取决于它的类型。

### 2.1 普通方法：先创建实例，再调

```python
class Dog:
    def __init__(self, name):
        self.name = name

    def bark(self):
        print(f"{self.name}：汪汪")

dog = Dog("旺财")    # ① 创建实例
dog.bark()           # ② 实例.方法()  →  旺财：汪汪
```

### 2.2 静态方法：直接用类名调

```python
class Math:
    @staticmethod
    def add(a, b):
        return a + b

Math.add(1, 2)   # 类名.方法()，不用创建实例
```

### 2.3 DDPM 中：`model(t)` 怎么就能调 `forward`？

PyTorch 的 `nn.Module` 实现了 `__call__`，让你可以像调函数一样调实例：

```python
model = SinusoidalPosEmb(dim=128)   # 创建实例
emb = model(t)                      # 等价于 model.forward(t)
```

普通 Python 类也能这样：

```python
class Adder:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):          # 实现 __call__
        return x + self.n

add5 = Adder(5)
add5(10)   # 15  —— 实例可以当函数用
```

### 速查

| 方法类型 | 定义 | 调用方式 | 需要实例？ |
|---------|------|---------|:---:|
| 普通方法 | `def foo(self, x)` | `obj.foo(x)` | ✅ |
| 静态方法 | `@staticmethod` + `def foo(x)` | `Cls.foo(x)` | ❌ |
| `__call__` | `def __call__(self, x)` | `obj(x)` | ✅ |

## 3. `__init__` —— 构造函数

创建对象时**自动调用**，用来设置初始状态。

```python
class Dog:
    def __init__(self, name, age):
        self.name = name    # 把参数存到实例上
        self.age = age

dog = Dog("旺财", 3)        # 自动调用 __init__
print(dog.name)             # 旺财
print(dog.age)              # 3
```

**不是** `dog.__init__("旺财", 3)`，而是 `Dog("旺财", 3)`，Python 自动把 `__init__` 转成构造调用。

## 4. `self` —— 指代"当前这个实例"

`self` 不是关键字，是**约定俗成的第一个参数名**。它指向调用方法的那个具体对象。

```python
class Dog:
    def __init__(self, name):
        self.name = name        # self.name 属于这个实例

    def bark(self):
        print(f"{self.name}：汪汪")   # 这里的 self 是调用时的那个狗

dog1 = Dog("旺财")
dog2 = Dog("来福")
dog1.bark()   # self = dog1 → "旺财：汪汪"
dog2.bark()   # self = dog2 → "来福：汪汪"
```

**一句话**：`self` = 谁调用这个方法，它就指谁。

## 5. `self.xxx` —— 实例属性

`self.xxx = 值` 把数据挂在实例身上，之后整个类的任何方法里都能通过 `self.xxx` 访问。

```python
class Counter:
    def __init__(self):
        self.count = 0          # 初始化实例属性

    def add(self, n):
        self.count += n         # 读写实例属性

    def show(self):
        print(self.count)       # 读取实例属性

c = Counter()
c.add(5)
c.add(3)
c.show()   # 8
```

## 6. `super().__init__()` —— 调用父类初始化

当你的类继承别人的类时，自己的 `__init__` 会**覆盖**掉父类的。必须手动调用 `super().__init__()` 让父类也完成初始化。

```python
class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)    # 让 Animal 设置 self.name
        self.breed = breed        # 再设置自己的

dog = Dog("旺财", "金毛")
print(dog.name)    # 旺财     ← 来自父类
print(dog.breed)   # 金毛     ← 来自自己
```

**不写 `super().__init__()` 的后果**：父类的属性不会被设置，访问会报错。

## 7. 实战对照：DDPM 中的类

```python
class SinusoidalPosEmb(nn.Module):          # 继承 nn.Module
    def __init__(self, dim: int):           # 构造时传入 dim
        super().__init__()                  # ① 让 nn.Module 初始化（必须）
        self.dim = dim                      # ② 把 dim 存为实例属性

    def forward(self, t):                   # self = 调用时的那个模型实例
        half_dim = self.dim // 2            # ③ 通过 self 读取 dim
        ...
        return emb
```

一步步拆解：

```python
model = SinusoidalPosEmb(dim=128)   # → 自动调用 __init__(self=model, dim=128)
                                    #   → super().__init__() 初始化 nn.Module 部分
                                    #   → self.dim = 128

emb = model(t)                      # → 自动调用 forward(self=model, t)
                                    #   → 内部 self.dim 就是 128
```

| 概念 | 你看到 | 实际发生了什么 |
|------|--------|---------------|
| `__init__(self, dim)` | 定义时有两个参数 | 调用时只传 `dim`，`self` 自动传入 |
| `self.dim = dim` | 赋值 | 把值存到**这个**模型实例上 |
| `super().__init__()` | 一行调用 | 让父类 `nn.Module` 完成初始化 |
| `self.dim // 2` | 读取属性 | 从实例上取出之前存的 `dim` |

## 8. 装饰器 —— `@xxx` 的本质

### 8.1 一句话理解

装饰器就是一个**包裹函数的函数**：你把函数 A 扔进去，它返回一个增强了的新函数。
`@` 只是语法糖，本质是**函数调用**。

### 8.2 无参数装饰器示例代码
#### 无参数 + 不修改返回值（纯增强功能）
最标准的日志/计时器，只加副作用，原样返回结果。

```python
import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)  # 执行原函数
        cost = time.perf_counter() - start
        print(f"[耗时] {func.__name__}: {cost:.3f}s")
        return result  # 原封不动返回
    return wrapper

@timer
def add(x, y):
    time.sleep(0.5)
    return x + y

print(add(3, 5))  # 输出耗时信息，然后打印 8（返回值没变）
```

#### 无参数 + 修改返回值（转换输出）
比如把返回的字符串**自动大写**，或者把数字**翻倍**。
```python
import functools

def double_result(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2  # 修改！把计算结果翻倍
    return wrapper

@double_result
def get_number():
    return 21

print(get_number())  # 输出 42（原函数返回21，被装饰器乘2了）
```



### 8.3 带参数的装饰器

有些装饰器自己也需要参数（如 `@lru_cache(maxsize=128)`）。这时需要**再包一层**。带参数的装饰器必须**三层嵌套**。
#### 有参数 + 不修改返回值（控制过程行为）
最经典的例子是**带重试机制的装饰器**：参数控制重试几次，但只要原函数最终成功，返回值必须原封不动地交给调用者

```python
import functools
import time
import random

def retry(max_attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    # 执行原函数，拿到结果
                    result = func(*args, **kwargs)
                    # 注意这里：只要不报错，直接 return 原结果，绝对不改
                    return result  
                except Exception as e:
                    last_exception = e
                    print(f"[重试] 第 {attempt} 次失败，{delay}s 后重试...")
                    time.sleep(delay)
            # 全部重试失败，抛出异常（注意，这里抛异常，相当于也没改返回值）
            raise last_exception
        return wrapper
    return decorator

# 使用：配置最多重试5次，每次间隔2秒
@retry(max_attempts=5, delay=2)
def unstable_api_call():
    if random.random() < 0.7:  # 70%概率失败
        raise ConnectionError("网络超时")
    return "成功数据"

# 调用者拿到的永远是原函数返回的 "成功数据"，装饰器只加了重试过程
data = unstable_api_call()  
print(data)  # 输出: 成功数据（返回值没变）
```

#### 有参数 + 修改返回值（带配置的转换）
给返回值**乘以指定倍数**，或加上指定**前后缀**。
```python
import functools

# 注意这是三层函数
def multiply(factor):          # 第一层：接收装饰器参数
    def decorator(func):       # 第二层：接收原函数
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # 第三层：接收原函数调用参数
            result = func(*args, **kwargs)
            return result * factor     # 修改！使用外部传入的 factor
        return wrapper
    return decorator

@multiply(factor=10)  # 带括号调用，传入参数
def get_score():
    return 8.5

print(get_score())  # 输出 85.0（原函数返回8.5，被装饰器乘了10）
```

### 8.4 常见内置装饰器速查

| 装饰器 | 作用 | 示例 |
|--------|------|------|
| `@staticmethod` | 方法不需要 `self` | `def add(a, b):` 无 self |
| `@classmethod` | 第一个参数是类 `cls` | `def from_str(cls, s):` |
| `@property` | 方法变属性，不加括号调用 | `obj.name` 而非 `obj.name()` |
| `@functools.lru_cache` | 缓存函数返回值 | `@lru_cache(maxsize=128)` |
| `@dataclass` | 自动生成 `__init__` | `class Point: x: int; y: int` |



