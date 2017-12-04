from abc import abstractmethod
from json import dump


class Training:
    def __init__(self, name):
        self.name = name
        self.q_model = None
        self.define_model()

    @abstractmethod
    def define_model(self):
        raise NotImplementedError

    def save_weights(self, path):
        self.q_model.save_weights(path)
        return print("The weights of your model saved.")

    def save_model(self, path, json=False):
        if json:
            model_json = self.q_model.to_json()
            with open(path, 'w') as f:
                dump(model_json, f)
            save_type = 'json'
        else:
            self.q_model.save(path)
            save_type = 'h5'
        return print("The model saved as {}.".format(save_type))
