"""
Contains variables and functions for simplifying regex code
"""

# save some common regex features as human readable variables
letter           = r'[a-zA-Z]'
double           = r'-?\d+\.\d+'
whitespace       = r'\s'
endline          = r'\n'
# zero or more whitespace (ws) followed by the regex feature 
ws_double  = r"\s*" + double    # for floats
ws_endline = r"\s*" + endline   # for newlines 


def maybe(string):
    """
    A regex wrapper for an arbitrary string.
    Allows a string to be present, but still matches if it is not present.
    """
    return r'(?:{:s})?'.format(string)

def two_or_more(string):
    """
    A regex wrapper for an arbitrary string.
    Allows an arbitrary number of successive valid matches to be matched.
    """
    return r'(?:{:s}){{2,}}'.format(string)


# a regex identifier for an xyz style geometry line, atom_label  x_coord   y_coord  z_coord
xyz_line_regex = r"[ \t]*" + letter + maybe(letter) + 3 * ws_double + ws_endline

# an xyz style geometry block of any size
xyz_block_regex = two_or_more(geom_line_regex)

