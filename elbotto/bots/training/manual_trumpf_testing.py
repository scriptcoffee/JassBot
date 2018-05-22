import numpy as np
from elbotto.bots.training.manual_testing import is_none, get_model, create_test_matrix


def manuel_test_input_predict(model=None, hand_cards=None, pushed=False, safe_in_textfile=False):
    hand_cards = is_none(hand_cards)

    t_model = get_model(model)

    input_matrix = create_test_matrix(hand_cards, pushed)

    if input_matrix is None:
        print("Your input is wrong. Please check your cards!")
    else:
        result = t_model.predict(np.asarray(input_matrix))
        print(
            'The prediction is: \n'
            ' hearts: {} \n diamonds: {} \n clubs: {} \n spades: {} \n OBEABE: {} \n UNDEUFE: {} \n SCHIEBE: {}'.format
            (result[0][0], result[0][1], result[0][2], result[0][3], result[0][4], result[0][5], result[0][6]))
        if safe_in_textfile:
            safe_as_txtfile(hand_cards, result)


def safe_as_txtfile(hand_cards, result, filename="result_Trumpfwahl.csv"):
    fobj_out = open(filename, "a+")
    fobj_out.write(
        'Karte1; Karte2; Karte3; Karte4; Karte5; Karte6; Karte7; Karte8; Karte9; '
        'Hearts; Diamonds; Clubs; Spades; Obeabe; Undeufe; Schieben; \n'
        '{}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}; \n'
            .format(hand_cards[0], hand_cards[1], hand_cards[2], hand_cards[3], hand_cards[4], hand_cards[5],
                    hand_cards[6], hand_cards[7], hand_cards[8],
                    result[0][0], result[0][1], result[0][2], result[0][3], result[0][4], result[0][5], result[0][6]
                    ))
    fobj_out.close()


# Set with 'model' that model you want to test.
# Input your hand cards into the list 'hand_cards'
# Set 'pushed' to True if your partner moved the trump decision to you.
# Set 'safe_in_textfile' if you want safe the results in a Textfile
if __name__ == '__main__':
    manuel_test_input_predict(model="./config/trumpf_network_model_final__2018-05-09_101936.h5",
                              hand_cards=["H6", "C8", "H9", "HJ", "D8", "D9", "D10", "S8", "S10"],
                              pushed=False, safe_in_textfile=True)
