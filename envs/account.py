import torch 

class VirtualAccount:
    def __init__(self, balance):
        self.balance = balance

    def withdraw(self, money):
        if type(money) == torch.Tensor:
            money = money.detach().numpy()
        if self.balance < money:
            return False
        else:
            self.balance -= money
            return True

    def deposit(self, money):
        if type(money) == torch.Tensor:
            money = money.detach().numpy()
        self.balance += money
        return True

    def get_balance(self):
        return self.balance

    def set_balance(self, new_balance):
        if type(new_balance) == torch.Tensor:
            new_balance = new_balance.detach().numpy()
        self.balance = new_balance