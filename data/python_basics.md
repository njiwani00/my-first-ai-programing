# Python Basics

Python is a popular programming language created by Guido van Rossum and first released in 1991. It is known for its clean, readable syntax, which uses indentation instead of curly braces to group code into blocks.

## Variables and Data Types

A variable is a name that refers to a value. You create one with the equals sign, for example age = 20. Python has several built-in data types: int for whole numbers, float for decimal numbers, str for text, and bool for True or False values.

## Lists and Dictionaries

A list stores an ordered collection of items in square brackets, for example fruits = ["apple", "banana", "mango"]. Lists are mutable, which means you can add, remove or change items after creating them. The first item in a list has index 0.

A dictionary stores key-value pairs in curly braces, for example marks = {"maths": 90, "physics": 85}. You look up a value by its key, as in marks["maths"]. A tuple is like a list but immutable, so it cannot be changed after it is created.

## Loops and Conditions

An if statement runs code only when a condition is true. You can add elif and else branches for other cases. A for loop repeats code for each item in a sequence, such as every element of a list or every number from range(5). A while loop repeats code as long as a condition stays true, so be careful to change the condition or it will loop forever.

## Functions

A function is a reusable block of code defined with the def keyword. For example, def greet(name): return "Hello " + name defines a function that takes one parameter. Functions help you avoid repeating code and make programs easier to test.

## Errors and Exceptions

When something goes wrong, Python raises an exception, such as a ZeroDivisionError or a KeyError. You can handle exceptions with a try and except block so the program does not crash.

## Virtual Environments and pip

A virtual environment is an isolated folder that holds the packages for one project, so different projects do not conflict. You create one with python -m venv .venv. The pip tool installs packages from the Python Package Index, for example pip install numpy.
