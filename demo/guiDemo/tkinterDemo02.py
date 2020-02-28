# -*- coding:UTF-8 -*-

from tkinter import *
import random


window = Tk()
window.title("GET RANDOM NUBER")
window.geometry('350x200')
lbl = Label(window,text = "Enter a number")
lbl.grid(column=0,row=0)
lb = Label(window)
txt = Entry(window,width=10)
txt.grid(column=1,row=0)
def clicked():
    number = range(1,int(txt.get()))
    lb.configure(text = "The random number is " + str(random.choices(number)))
    lb.grid(column=0,row=1)
btn = Button(window,fg="black",bg="green",text="Start",command=clicked)
btn.grid(column=10,row=0)
window.mainloop()
