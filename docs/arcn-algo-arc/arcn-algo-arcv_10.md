# 栈和队列

> 原文：[`www.algorithm-archive.org/contents/stacks_and_queues/stacks_and_queues.html`](https://www.algorithm-archive.org/contents/stacks_and_queues/stacks_and_queues.html)

栈和队列是计算机科学中同一枚硬币的两面。它们都是简单的数据结构，可以存储多个元素，但一次只能使用一个元素。这两种结构之间最大的区别是你可以在数据结构中访问元素的方式。

在*栈*中，数据遵循*后进先出*（LIFO）原则，这基本上意味着你最后放入的元素将是第一个被取出的元素。它就像现实生活中的栈一样。如果你在一摞书上放一本书，当你翻看这摞书时，你最先看到的是你刚刚放上去的那本书。

在*队列*中，数据遵循*先进先出*（FIFO）原则，这意味着你最先放入的元素将是第一个被取出的元素。想象一下排队的人群。如果第一个排队买 groceries 的人不是第一个得到服务员注意的人，那就太不公平了。

然而，大多数情况下，队列和栈被同等对待。必须有一种方法：

1.  查看第一个元素（`top()`）

1.  移除第一个元素（`pop()`）

1.  向数据结构中推入元素（`push()`）

这种表示法取决于你使用的语言。例如，队列通常会使用 `dequeue()` 而不是 `pop()`，以及 `front()` 而不是 `top()`。你将在本书算法的源代码中看到语言特定的细节，所以现在重要的是要知道栈和队列是什么，以及如何访问它们持有的元素。

## 示例代码

这里是一个栈的简单实现：

```
interface IStack<T> {
  /**
   * `pop` removes last element from the stack and returns the same
   */
  pop(): T;
  /**
   * `push` adds element to last of the stack and returns the size
   */
  push(data: T): number;
  /**
   * `size` return size or length of the stack
   */
  size(): number;
  /**
   * `top` returns last element of the stack
   */
  top(): T;
}

class Stack<T> implements IStack<T> {
  private readonly list: Array<T> = [];

  public push(data: T) {
    return this.list.push(data);
  }

  public pop() {
    return this.list.pop();
  }

  public size() {
    return this.list.length;
  }

  public top() {
    return this.list[this.list.length - 1];
  }
}

function exampleStack() {
  const numberStack = new Stack<number>();

  numberStack.push(4);
  numberStack.push(5);
  numberStack.push(9);

  console.log(numberStack.pop());
  console.log(numberStack.size());
  console.log(numberStack.top());
}

exampleStack(); 
```

```
import java.util.List;
import java.util.ArrayList;

public class StackTest {

    public static void main(String[] args) {
    IStack<Integer> intStack = new Stack<>();

    intStack.push(4);
    intStack.push(5);
    intStack.push(9);

    System.out.println(intStack.pop());
    System.out.println(intStack.size());
    System.out.println(intStack.top());
    }

}

interface IStack<T> {
   /*
    * 'pop' removed the last element from the stack and returns it
    */
    T pop();

   /*
    * 'push' adds an element to at the end of the stack and returns the new size
    */
    int push(T element);

   /*
    * 'size' returns the length of the stack
    */
    int size();

   /*
    * 'top' returns the first element of the stack
    */
    T top();
}

class Stack<T> implements IStack<T> {

    private List<T> list;

    public Stack() {
        this.list = new ArrayList<>();
    }

    public T pop() {
        return this.list.remove(this.size() - 1);
    }

    public int push(T element) {
        this.list.add(element);
        return this.size();
    }

    public int size() {
        return this.list.size();
    }

    public T top() {
        return this.list.get(this.size() - 1);
    }

} 
```

```
#include<iostream>
#include<cassert>
#include<memory>

namespace my {
    /**
     * implementation using linked list
     * [value][next] -> [value][next] -> ... -> [value][next]
     * (top Node)      (intermediat Nodes)
     * left most Node represents top element of stack
     */
    template<typename T>
    struct Node {
        /**
        * next: will store right Node address
        */
        T value;
        std::unique_ptr<Node<T>> next;
        Node(const T& V) : value(V) { }
    };

    template<typename T>
    class stack {
    private:
        /**
         * top_pointer: points to left most node
         * count: keeps track of current number of elements present in stack
         */
        std::unique_ptr<Node<T>> top_pointer;
        size_t count;
    public:
        stack() : count(0ULL) { }

        void push(const T& element) {
            auto new_node = std::make_unique<Node<T>>(element);
            new_node->next = std::move(top_pointer);
            top_pointer = std::move(new_node);
            count = count + 1;
        }

        void pop() {
            if (count > 0) {
                top_pointer = std::move(top_pointer->next);
                count = count - 1;
            }
        }

        T& top() {
            assert(count > 0 and "calling top() on an empty stack");
            return top_pointer->value;
        }
        // returning mutable reference can very be usefull if someone wants to modify top element

        T const& top() const {
            assert(count > 0 and "calling top() on an empty stack");
            return top_pointer->value;
        }

        size_t size() const { return count; }

        bool empty() const { return count == 0; }

        ~stack() {
            while (top_pointer.get() != nullptr) {
                top_pointer = std::move(top_pointer->next);
            }
        }
    };
}

int main() {
  my::stack<int> intStack;

  intStack.push(4);
  intStack.push(5);
  intStack.push(9);

  int topElement = intStack.top();
  intStack.pop();
  std::cout << topElement << '\n';
  std::cout << intStack.size() << '\n';
  std::cout << intStack.top() << '\n';
  return 0;
} 
```

```
struct Stack<T> {
    list: Vec<T>
}

impl<T> Stack<T> {
    fn new() -> Self {
        Stack {
            list: Vec::new(),
        }
    }

    // Note that this returns a reference to the value
    // This is in contrast to pop which gives ownership of the value
    fn top(&self) -> Option<&T> {
        self.list.last()
    }

    fn pop(&mut self) -> Option<T> {
        self.list.pop()
    }

    fn push(&mut self, item: T) {
        self.list.push(item);
    }

    fn size(&self) -> usize {
        self.list.len()
    }
}

fn main() {
    let mut i32stack: Stack<i32> = Stack::new();

    i32stack.push(4);
    i32stack.push(5);
    i32stack.push(6);

    println!("{:?}", i32stack.pop().unwrap()); // 6
    println!("{:?}", i32stack.size()); // 2
    println!("{:?}", i32stack.top().unwrap()); // 5
} 
```

```
#!/usr/bin/env python3

__author__ = "Michael Ciccotosto-Camp"

from typing import TypeVar, Generic

T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self.__list: list[T] = []

    def pop(self) -> T:
        return self.__list.pop()

    def push(self, element: T) -> int:
        self.__list.append(element)
        return len(self)

    def top(self) -> T:
        return self.__list[-1]

    def __len__(self) -> int:
        return len(self.__list)

    def __str__(self) -> str:
        return str(self.__list)

def main() -> None:
    int_stack: Stack[int] = Stack()

    int_stack.push(4)
    int_stack.push(5)
    int_stack.push(9)

    print(int_stack.pop())
    print(len(int_stack))
    print(int_stack.top())

if __name__ == "__main__":
    main() 
```

这里是一个队列的简单实现：

```
interface IQueue<T> {
  /**
   * `dequeue` removes first element from the queue and returns the same
   */
  dequeue(): T;
  /**
   * `enqueue` adds element to last of the queue and returns the size
   */
  enqueue(data: T): number;
  /**
   * `size` return size or length of the queue
   */
  size(): number;
  /**
   * `front` returns first element of the queue
   */
  front(): T;
}

class Queue<T> implements IQueue<T> {
  private readonly list: Array<T> = [];

  public enqueue(data: T) {
    return this.list.push(data);
  }

  public dequeue() {
    return this.list.shift();
  }

  public size() {
    return this.list.length;
  }

  public front() {
    return this.list[0];
  }
}

function exampleQueue() {
  const numberQueue = new Queue<number>();

  numberQueue.enqueue(4);
  numberQueue.enqueue(5);
  numberQueue.enqueue(9);

  console.log(numberQueue.dequeue());
  console.log(numberQueue.size());
  console.log(numberQueue.front());
}

exampleQueue(); 
```

```
import java.util.List;
import java.util.ArrayList;

public class QueueTest {

    public static void main(String[] args) {
    IQueue<Integer> intQueue = new Queue<>();

    intQueue.enqueue(4);
    intQueue.enqueue(5);
    intQueue.enqueue(9);

    System.out.println(intQueue.dequeue());
    System.out.println(intQueue.size());
    System.out.println(intQueue.front());
    }

}

interface IQueue<T> {

   /*
    * 'dequeue' removes the first element from the queue and returns it
    */
    T dequeue();

   /*
    * 'enqueue' adds an element at the end of the queue and returns the new size
    */
    int enqueue(T element);

   /*
    * 'size' returns the size of the queue
    */
    int size();

   /*
    * 'front' returns the first element of the queue without removing it
    */
    T front();
}

class Queue<T> implements  IQueue<T> {

    private List<T> list;

    public Queue() {
        this.list = new ArrayList<>();
    }

    public T dequeue() {
        return this.list.remove(0);
    }

    public int enqueue(T element) {
        this.list.add(element);
        return this.size();
    }

    public int size() {
        return this.list.size();
    }

    public T front() {
        return this.list.get(0);
    }

} 
```

```
#include<iostream>
#include<memory>
#include<cassert>

namespace my {
    /**
     * implementation using linked list
     * [value][next] -> [value][next] -> ... -> [value][next]
     *  (front Node)   (intermediat Nodes)     (rear Node)
     */
    template<typename T>
    struct Node {
        /**
        * next: will store right Node address
        */
        T value;
        std::shared_ptr<Node<T>> next;
        Node(const T& V) : value(V) { }
    };

    template<typename T>
    class queue {
    private:
        /**
         * front_pointer:  points to left most node
         * count: keeps track of current number of elements present in queue
         * rear_pointer:  points to most recent Node added into the queue, which is right most Node
         */
        std::shared_ptr<Node<T>> front_pointer;
        std::shared_ptr<Node<T>> rear_pointer;
        size_t count;
    public:
        queue() : count(0ULL) { }

        void enqueue(const T& element) {
            auto new_node = std::make_shared<Node<T>>(element);
            if (count > 0) {
                rear_pointer->next = new_node;
                rear_pointer = new_node;
            } else {
                rear_pointer = front_pointer = new_node;
            }
            count = count + 1;
        }

        void dequeue() {
            if (count > 1) {
                front_pointer = front_pointer->next;
                count = count - 1;
            } else if (count == 1) {
                front_pointer.reset();
                rear_pointer.reset();
                count = count - 1;
            }
        }

        T& front() {
            assert(count > 0 && "calling front on an empty queue");
            return front_pointer->value;
        }

        T const& front() const {
            assert(count > 0 && "calling front on an empty queue");
            return front_pointer->value;
        }

        size_t size() const { return count; }

        bool empty() const { return count == 0; }

        ~queue() {
            while (front_pointer.get() != nullptr) {
                front_pointer = front_pointer->next;
            }
        }
    };
}

int main() {
  my::queue<int> intQueue;
  intQueue.enqueue(4);
  intQueue.enqueue(5);
  intQueue.enqueue(9);

  int frontElement = intQueue.front();
  intQueue.dequeue();
  std::cout << frontElement << '\n';
  std::cout << intQueue.size() << '\n';
  std::cout << intQueue.front() << '\n';
  return 0;
} 
```

```
use std::collections::VecDeque;

struct Queue<T> {
    list: VecDeque<T>
}

impl<T> Queue<T> {
    fn new() -> Self {
       Queue{
           list: VecDeque::new(),
       }
    }

    // Note that this returns a reference to the value
    // This is in contrast to dequeue which gives ownership of the value
    fn front(&self) -> Option<&T> {
        self.list.front()
    }

    fn dequeue(&mut self) -> Option<T> {
        self.list.pop_front()
    }

    fn enqueue(&mut self, item: T) {
        self.list.push_back(item);
    }

    fn size(&self) -> usize {
        self.list.len()
    }
}

fn main() {
    let mut i32queue = Queue::new();

    i32queue.enqueue(4);
    i32queue.enqueue(5);
    i32queue.enqueue(6);

    println!("{:?}", i32queue.dequeue().unwrap()); // 4
    println!("{:?}", i32queue.size()); // 2
    println!("{:?}", i32queue.front().unwrap()); // 5
} 
```

```
#!/usr/bin/env python3

__author__ = "Michael Ciccotosto-Camp"

from typing import TypeVar, Generic

T = TypeVar("T")

class Queue(Generic[T]):
    def __init__(self) -> None:
        self.__list: list[T] = list()

    def dequeue(self) -> T:
        return self.__list.pop(0)

    def enqueue(self, element: T) -> int:
        self.__list.append(element)
        return len(self)

    def front(self) -> T:
        return self.__list[0]

    def __len__(self) -> int:
        return len(self.__list)

    def __str__(self) -> str:
        return str(self.__list)

def main() -> None:
    int_queue: Queue[int] = Queue()

    int_queue.enqueue(4)
    int_queue.enqueue(5)
    int_queue.enqueue(9)

    print(int_queue.dequeue())
    print(len(int_queue))
    print(int_queue.front())

if __name__ == "__main__":
    main() 
```

## 许可证

##### 代码示例

代码示例受 MIT 许可证（在 [LICENSE.md](https://github.com/algorithm-archivists/algorithm-archive/blob/main/LICENSE.md) 中找到）许可。

##### 文本

本章的文本由 [James Schloss](https://github.com/leios) 编写，并受 [Creative Commons Attribution-ShareAlike 4.0 国际许可协议](https://creativecommons.org/licenses/by-sa/4.0/legalcode) 许可。

[](https://creativecommons.org/licenses/by-sa/4.0/)

![](https://creativecommons.org/licenses/by-sa/4.0/)

##### 拉取请求

在初始许可后（[#560](https://github.com/algorithm-archivists/algorithm-archive/pull/560)），以下拉取请求已修改本章的文本或图形：

+   none
