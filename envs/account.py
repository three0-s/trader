class VirtualAccount:
    def __init__(self, balance):
        self.balance = balance

    def withdraw(self, money):
        if self.balance < money:
            return False
        else:
            self.balance -= money
            return True

    def deposit(self, money):
        self.balance += money
        return True

    def get_balance(self):
        return self.balance

    def set_balance(self, new_balance):
        self.balance = new_balance