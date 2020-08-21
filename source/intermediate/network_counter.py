class __metadata:
    rules = [
        "The list should only contain the numbers when the "
        "`io_bound_state` of the network counter is `False`"
    ]
    answers = ["not locals()['.0'].io_bound_state"]


import random


class NetworkCounter:
    def __init__(self, stop):
        self.counter = 0
        self.stop = stop
        self.io_bound_state = False
        self.already_sent_packages = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter == self.stop:
            print("Already sent these packages: ", self.already_sent_packages)
            raise StopIteration
        self.counter += 1
        if random.randint(0, 3) == 1:
            self.io_bound_state ^= 1
        if self.io_bound_state:
            self.already_sent_packages.append(self.counter)
        return self.counter


print(
    "Sending these packages: ",
    [package for package in NetworkCounter(10) if ...],
)
