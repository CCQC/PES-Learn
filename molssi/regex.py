"""
Contains variables and functions for simplifying regex code
"""

# save some common regex features as human readable variables
letter           = r'[a-zA-Z]'
double           = r'-?\d+\.\d+'
integer          = r'\d+'
whitespace       = r'\s'
endline          = r'\n'
# zero or more whitespace (ws) followed by the regex feature 
ws_double  = r'\s*' + double 
ws_endline = r'\s*' + endline 
ws_int     = r'\s*' + integer 


def maybe(string):
    """
    A regex wrapper for an arbitrary string.
    Allows a string to be present, but still matches if it is not present.
    """
    return r'(?:{:s})?'.format(string)

def one_or_more(string):
    """
    A regex wrapper for an arbitrary string.
    Allows an arbitrary number of successive valid matches (at least one) to be matched.
    """
    return r'(?:{:s}){{1,}}'.format(string)

def two_or_more(string):
    """
    A regex wrapper for an arbitrary string.
    Allows an arbitrary number of successive valid matches (at least two) to be matched.
    """
    return r'(?:{:s}){{2,}}'.format(string)


# a regex identifier for an xyz style geometry line, atom_label  x_coord   y_coord  z_coord
xyz_line_regex = r'[ \t]*' + letter + maybe(letter) + 3 * ws_double + ws_endline
# an xyz style geometry block of any size
xyz_block_regex = two_or_more(xyz_line_regex)

# define compact internal coordinates regex identifier
# e.g.
# O
# H 1 1.0
# H 1 1.0 2 104.5
# H 1 1.0 2 100.00 3 180.0

intcocompact_1 = letter + maybe(letter) + ws_endline
intcocompact_2 = letter + maybe(letter) + ws_int + ws_double + ws_endline
intcocompact_3 = letter + maybe(letter) + ws_int + ws_double + ws_int + ws_double + ws_endline
intcocompact_4 = letter + maybe(letter) + ws_int + ws_double + ws_int + ws_double + ws_int + ws_double + ws_endline

intcoords_compact_regex = intcocompact_1 + intcocompact_2 + maybe(intcocompact_3) + maybe(one_or_more(intcocompact_4))


