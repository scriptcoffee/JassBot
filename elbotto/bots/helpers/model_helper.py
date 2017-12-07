from json import dump
from datetime import datetime


def save_weights(model, path):
    model.save_weights(path)
    return print("The weights of your model saved.")


def save_model(model, path, json=False):
    if json:
        model_json = model.to_json()
        with open(path, 'w') as f:
            dump(model_json, f)
        save_type = 'json'
    else:
        model.save(path)
        save_type = 'h5'
    return print("The model saved as " + save_type + ".")


def save_weights_and_model(model, game_counter, log_dir, model_name):
        file_addition = "_" + str(game_counter) + datetime.now().strftime("__%Y-%m-%d_%H%M%S")

        save_model(model, "{}{}_model{}.h5".format(log_dir, model_name, file_addition))
        save_model(model, "{}{}_model{}.json".format(log_dir, model_name, file_addition), True)
        save_weights(model, "{}{}_weights{}.h5".format(log_dir, model_name, file_addition))