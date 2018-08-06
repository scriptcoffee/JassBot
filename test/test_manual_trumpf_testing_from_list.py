import pytest
from openpyxl.cell.cell import Cell
from elbotto.bots.training.manual_trumpf_testing_from_list import matrix_to_list, \
                                                                    calc_prediction_matrix, \
                                                                    shrink_lines_to_handcards, \
                                                                    prepared_prediction_matrix, \
                                                                    extract_lines_from_sheet, \
    manuel_test_list_input_predict, create_table_title, safe_as_excelfile, fillup_table


@pytest.mark.parametrize("test_input, expected", [
    ("", None),
    (None, None),
    (['A1', 'A2', 'A3', 'A4', 'A5', 'A6'], None),
    (('A1', 'A2', 'A3', 'A4', 'A5', 'A6'), None),
    ((('A1', 'A2', 'A3', 'A4', 'A5'), ('B1', 'B2', 'B3', 'B4', 'B5'), ('C1', 'C2', 'C3', 'C4', 'C5')), None)

])
@pytest.mark.statistical
def test_extract_lines_from_sheet_wrong_inputs(test_input, expected):
    assert extract_lines_from_sheet(test_input) == expected


@pytest.mark.statistical
def test_extract_lines_from_sheet_empty_list():
    input_list = (Cell('Card1'), Cell('Card2'), Cell('Card3'), Cell('Card4'), Cell('Card5'),
                  Cell('Card6'), Cell('Card7'), Cell('Card8'), Cell('Card9'), Cell('Moved?'))
    assert extract_lines_from_sheet(input_list) is None


# @pytest.mark.statistical
# def test_extract_lines_from_sheet_correct_input():
#     input_list = {(Cell('Card1'), Cell('Card2'), Cell('Card3'), Cell('Card4'), Cell('Card5'),
#                   Cell('Card6'), Cell('Card7'), Cell('Card8'), Cell('Card9'), Cell('Moved?')),
#
#                   (Cell('H6'), Cell('C8'), Cell('H9'), Cell('HJ'), Cell('D8'),
#                   Cell('D9'), Cell('D10'), Cell('S8'), Cell('S10'), Cell('')),
#
#                   (Cell('H6'), Cell('C8'), Cell('H9'), Cell('HJ'), Cell('D8'),
#                   Cell('D9'), Cell('D10'), Cell('S8'), Cell('S10'), Cell('x'))
#                   }
#     output = [['H6', 'C8', 'H9', 'HJ', 'D8', 'D9', 'D10', 'S8', 'S10', False],
#               ['H6', 'C8', 'H9', 'HJ', 'D8', 'D9', 'D10', 'S8', 'S10', True]]
#     print('Hello Test1')
#     result = extract_lines_from_sheet(input_list)
#     print('Hello test2')
#     print('INPUT VALUE: {0}'.format(input_list[1][0].value))
#     print('RESULT: {0}'.format(result))
#
#     assert result == output

