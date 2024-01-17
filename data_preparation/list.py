import os
import random

path = ("WHU256/A/")
list = os.listdir(path)
random.shuffle(list)
a = open("WHU256/list/train.txt", "w")
b = open("WHU256/list/test.txt", "w")
c = open("WHU256/list/val.txt", "w")
sum = 0

# 8:1:1
for i in range(6096):
    a.write(list[i] + "\r")
    sum = sum + 1
for i in range(6096, 6858):
    b.write(list[i] + "\r")
    sum = sum + 1
for i in range(6858, 7620):
    c.write(list[i] + "\r")
    sum = sum + 1
print(sum)