#!/usr/bin/python -tt
# Copyright 2010 Google Inc.
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

# Google's Python Class
# http://code.google.com/edu/languages/google-python-class/

def read_file(filename) :
    f = open(filename, "r")
    for line in f:
        for str in line.split():
            yield str

# Calls the above functions with interesting inputs.
def main():
    for word in read_file("small.txt"):
        print(word)


if __name__ == '__main__':
    main()
