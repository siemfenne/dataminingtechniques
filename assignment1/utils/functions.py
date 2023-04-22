import math

def print_section(section: str, width: int = 140):
    """ 
    Print a certain text in the format of:
    %%%%%%%%%%%%%%%%%
    %%% {section} %%%
    %%%%%%%%%%%%%%%%%
    """
    section_len = len(section)
    left_len = (width - section_len - 2) // 2
    right_len = width - left_len - section_len - 2
    string = \
        width * "%" + "\n" + \
        left_len * "%" + " " + section + " " + right_len * "%" + "\n" + \
        width * "%"
    print(string)