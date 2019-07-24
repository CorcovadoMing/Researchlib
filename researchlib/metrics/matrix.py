import torch


class Matrix:
    def __init__(self):
        pass

    def forward(self, loss_input):
        pass

    def forward_g(self, loss_input):
        pass

    def forward_d(self, loss_input):
        pass

    def output(self):
        pass

    def reset(self):
        pass


# TODO
#
# class TestMatrix:
#     def __init__(self, f, name):
#         self.total = 0
#         self.value = 0
#         self.f = f
#         self.name = name

#     def forward(self, loss_input):
#         with torch.no_grad():
#             y_pred, y_true = loss_input[0], loss_input[1]
#             self.value += self.f(y_pred, y_true)
#             self.total += y_pred.size(0)

#     def output(self):
#         value = (float(self.value) / float(self.total))
#         return {self.name: value}

#     def reset(self):
#         self.value = 0
#         self.total = 0
