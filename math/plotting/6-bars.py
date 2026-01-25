#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# your code here
plt.figure()
people = ['Farrah', 'Fred', 'Felicia']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
width = 0.5

x = np.arange(len(people))
bottom = np.zeros(len(people))

for i in range(len(fruit)):
    plt.bar(x, fruit[i], width, bottom=bottom, color=colors[i],
            label=fruit_names[i])
    bottom += fruit[i]

plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.xticks(x, people)
plt.yticks(range(0, 81, 10))
plt.ylim(0, 80)
plt.legend()
plt.show()
