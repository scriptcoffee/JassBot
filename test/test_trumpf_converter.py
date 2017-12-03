from elbotto.bots.training.trumpf_converter import Trumpf, TrumpfCard, TrumpfColor
from elbotto.bots.training.trumpf_converter import trumpf_converter, has_code_valid_format


def test_class_trumpfcolof_create():
    trumpfcolor = TrumpfColor("RAINBOW", 20)

    assert trumpfcolor.name == "RAINBOW"
    assert trumpfcolor.value == 20


def test_class_trumpfcard_create():
    trumpf = TrumpfCard("JOKER", "RAINBOW", 20)

    assert trumpf.mode == "JOKER"
    assert trumpf.trumpf_color.name == "RAINBOW"
    assert trumpf.trumpf_color.value == 20


def test_has_code_valid_format_string():
    assert has_code_valid_format("test") is True


def test_has_code_valid_format_int():
    assert has_code_valid_format(15) is True


def test_has_code_valid_format_none():
    assert has_code_valid_format(None) is False


def test_has_code_valid_format_list():
    assert has_code_valid_format([]) is False


def test_trumpf_converter_by_number():
    trumpf_list = [["TRUMPF", Trumpf.DIAMONDS.name, Trumpf.DIAMONDS.value],
                   ["TRUMPF", Trumpf.HEARTS.name, Trumpf.HEARTS.value],
                   ["TRUMPF", Trumpf.SPADES.name, Trumpf.SPADES.value],
                   ["TRUMPF", Trumpf.CLUBS.name, Trumpf.CLUBS.value],
                   ["OBEABE", None],
                   ["UNDEUFE", None],
                   ["SCHIEBE", None]]

    for i in range(6):
        trumpf = trumpf_converter(i)

        assert trumpf.mode == trumpf_list[i][0]
        if trumpf_list[i][1] is not None:
            assert trumpf.trumpf_color.name == trumpf_list[i][1]
            assert trumpf.trumpf_color.value == trumpf_list[i][2]
        else:
            assert trumpf.trumpf_color.name == ""
            assert trumpf.trumpf_color.value == ""


def test_trumpf_converter_by_name():
    trumpf_list = [["TRUMPF", Trumpf.DIAMONDS.name, Trumpf.DIAMONDS.value],
                   ["TRUMPF", Trumpf.HEARTS.name, Trumpf.HEARTS.value],
                   ["TRUMPF", Trumpf.SPADES.name, Trumpf.SPADES.value],
                   ["TRUMPF", Trumpf.CLUBS.name, Trumpf.CLUBS.value],
                   ["OBEABE", None],
                   ["UNDEUFE", None],
                   ["SCHIEBE", None]]

    trumpf_names = ['DIAMONDS',
                    'HEARTS',
                    'SPADES',
                    'CLUBS',
                    'OBEABE',
                    'UNDEUFE',
                    'SCHIEBE']

    for i in range(len(trumpf_names)):
        trumpf = trumpf_converter(trumpf_names[i])

        assert trumpf.mode == trumpf_list[i][0]
        if trumpf_list[i][1] is not None:
            assert trumpf.trumpf_color.name == trumpf_list[i][1]
            assert trumpf.trumpf_color.value == trumpf_list[i][2]
        else:
            assert trumpf.trumpf_color.name == ""
            assert trumpf.trumpf_color.value == ""


def test_trumpf_converter_convert_color_code():
    trumpf_color_code_list_from_logs = [[0, "DIAMONDS"],
                                        [1, "HEARTS"],
                                        [2, "SPADES"],
                                        [3, "CLUBS"]]

    trumpf_color_code_list_from_server = [[0, "HEARTS"],
                                          [1, "DIAMONDS"],
                                          [2, "CLUBS"],
                                          [3, "SPADES"]]

    for i in range(len(trumpf_color_code_list_from_logs)):
        trumpf = trumpf_converter(i)
        j = trumpf.trumpf_color.value

        assert trumpf_color_code_list_from_logs[i][1] == trumpf_color_code_list_from_server[j][1]


def test_trumpf_converter_not_in_list():
    assert trumpf_converter("ACRON") is None


def test_trumpf_converter_to_high_number():
    assert trumpf_converter(7) is None
