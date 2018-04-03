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

# define generalized compact internal coordinates regex identifier
# e.g.
# O
# H 1 1.0
# H 1 1.0 2 104.5
# H 1 1.0 2 100.00 3 180.0

intcocompact_1 = r'[ \t]*' + letter + maybe(letter) + ws_endline
intcocompact_2 = r'[ \t]*' + letter + maybe(letter) + ws_int + ws_double + ws_endline
intcocompact_3 = r'[ \t]*' + letter + maybe(letter) + ws_int + ws_double + ws_int + ws_double + ws_endline
intcocompact_4 = r'[ \t]*' + letter + maybe(letter) + ws_int + ws_double + ws_int + ws_double + ws_int + ws_double + ws_endline

# assume at least two atoms
intcoords_compact_regex = intcocompact_1 + intcocompact_2 + maybe(intcocompact_3) + maybe(one_or_more(intcocompact_4))

# define generalized standard internal coordinates regex identifier
# e.g.
# O
# H 1 ROH
# H 1 R2 2 AHOH
# H 1 R3 2 A2 3 D1
# ...
# ROH   = 1.0
# R2    = 1.1
# R3    = 1.2
# AHOH  = 100.5
# A2    = 90.0
# D1    = 120.00

coord_label = one_or_more(letter) + maybe(one_or_more(integer))
ws_coord_label = r'\s*' + coord_label

# zero or more spaces/tabs, atom label, connectivity and coordinate labels 
intco_1 = r'[ \t]*' + letter + maybe(letter) + ws_endline
intco_2 = r'[ \t]*' + letter + maybe(letter) + ws_int + ws_coord_label + ws_endline
intco_3 = r'[ \t]*' + letter + maybe(letter) + ws_int + ws_coord_label + ws_int + ws_coord_label + ws_endline
intco_4 = r'[ \t]*' + letter + maybe(letter) + ws_int + ws_coord_label + ws_int + ws_coord_label + ws_int + ws_coord_label + ws_endline

# assume at least two atoms
intcoords_regex = intco_1 + intco_2 + maybe(intco_3) + maybe(one_or_more(intco_4))

# this function should go in some class later
def find_internal_coordinates(filestring):
    """
    Finds the internal coordinates in a file string and creates a dictionary of geometry parameters and corresponding values
    """
    # find internal coordinate definition block
    intcoord = re.findall(intcoords_regex, filestring)[0]
    # isolate atom labels and geometry parameter labels
    # atom labels will always be at index 0, 1, 3, 6, 6++4...
    tmp = re.findall(coord_label, intcoord)
    atoms = []
    for i, s in enumerate(tmp):
            if (i == 0) or (i == 1) or (i == 3):
                atoms.append(tmp[i])
            if ((i >= 6) and ((i-6) % 4 == 0)):
                atoms.append(tmp[i])
    parameters = [x for x in tmp if x not in atoms]
    
    # create a dictionary of geometry parameters without values set
    coord_dict = {key: None for key in parameters}
    # and find the values of these geometry parameters in the file
    for label in coord_dict:
        match = "{}\s*=\s*(-?\d+\.\d+".format(label)
        tmp = re.findall(match, inp)
        if len(tmp) > 1:
            raise Exception("More than one definition for internal coordinate parameter {} found".format(label))
        if len(tmp) == 0:
            raise Exception("Geometry parameter {} not defined in the input".format(label))
        else:
            value = float(tmp[0])
            coord_dict[label] = value
    return coord_dict
