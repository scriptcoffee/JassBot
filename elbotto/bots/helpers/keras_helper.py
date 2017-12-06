import numpy as np
from json import dump
from elbotto.card import Card


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


def prepare_game_input(input_layer_size, game_type, hand_cards, table_cards, played_cards):
    trumpf_offset = input_layer_size - 6
    inputs = np.zeros((input_layer_size,))

    for hand_card in hand_cards:
        inputs[hand_card.id] = 1

    for i in range(len(table_cards)):
        table_card = table_cards[i]
        table_card = Card.create(table_card["number"], table_card["color"])
        input_index = (i + 1) * 36 + table_card.id
        inputs[input_index] = 1

    for played_card in played_cards:
        input_index = 4 * 36 + played_card.id
        inputs[input_index] = 1

    if game_type.mode == "TRUMPF":
        inputs[game_type.trumpf_color.value + trumpf_offset] = 1
    elif game_type.mode == "OBEABE":
        inputs[trumpf_offset + 4] = 1
    elif game_type.mode == "UNDEUFE":
        inputs[trumpf_offset + 5] = 1

    inputs = np.reshape(inputs, (1, input_layer_size))

    return inputs
