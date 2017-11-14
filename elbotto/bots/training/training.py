from json import dump


class Training(object):
    def __init__(self, name):
        self.name = name
        self.q_model = None
        self.define_model()

    def define_model(self):
        pass

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
        return print("The model saved as " + save_type + ".")

    @staticmethod
    def create_input():
        pass

    @staticmethod
    def create_target():
        pass

    def train_the_model(self):
        pass
