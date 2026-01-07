""" Implementation from scratch of a morphological lexicon (like lexc)
    with or without replace rules (like xfst)

    Note that this is not a final state transducer, so this can only be used for
    toy lexicons without several thousands or millions of possible word forms
"""

# Common library code

import re
from collections import defaultdict
import random

MAX_RECURSION_DEPTH = 3 # Max number of times to enter the same sublexicon during processing of one word
                        # This avoids long processing times, huge memory consumption and stack overflow

depth = defaultdict(int) # Keep track of recursion depth for each sublexicon

wordform2analysis = defaultdict(set) # Lookup word form -> analysis
analysis2wordform = defaultdict(set) # Lookup analysis -> word form

class State:
    """ This object keeps track of the state of one particular attempt to
        traverse the network """
    def __init__(self, input, output, tail, is_analyzer, return_all):
        self._input = input              # Input string built so far
        self._output = output            # Output string built so far
        self._tail = tail                # Tail that needs to match any remaining input
        self._is_analyzer = is_analyzer  # True: is an analyzer, False: is a generator
        self._return_all = return_all    # True: return all output strings in lexicon, 
                                         # False: return just matching words
        self._result = []                # Collect all input, output pairs of successful analyses here
        
    def get_input(self):
        return self._input

    def get_output(self):
        return self._output

    def get_tail(self):
        return self._tail

    def is_analyzer(self):
        return self._is_analyzer

    def add_result(self, result):
        self._result.extend(result)

    def get_result(self):
        return self._result

    def return_all(self):
        return self._return_all

class ReplacedString:
    """ We need an object that stores a string that can be updated
        and implicitly returned from a function """

    def __init__(self, string, debug=False):
        self._string = string
        self._debug = debug

    def set_string(self, string):
        self._string = string

    def get_string(self):
        return self._string
    
    def is_debug(self):
        return self._debug
    

def entry_t(input, output, continuation, state):
    """ Process an "arc" in the network
        entry_t stands for transducer: input and output may be different
    """

    # Swap input and output sides if we are dealing with a generator rather than an analyzer
    if state.is_analyzer():
        i, o = input, output
    else:
        i, o = output, input

    # Consume input if possible and continue to continuation lexicon
    if state.get_tail().startswith(i) or state.return_all():
        new_state = State(state.get_input() + i, state.get_output() + o, 
                          state.get_tail()[len(i):], state.is_analyzer(), state.return_all())
        if continuation is None:
            if len(new_state.get_tail()) == 0:
                # Successfully analyzed a full word form
                state.add_result([(new_state.get_input(), new_state.get_output())])
            # Else: failure
        elif depth[continuation] < MAX_RECURSION_DEPTH:
            depth[continuation] += 1
            continuation(new_state)
            state.add_result(new_state.get_result())
            depth[continuation] -= 1


def entry_a(input, continuation, state):
    """ Syntactic sugar for entry_t, when input and output sides are the same
        entry_a stands for automaton
    """
    entry_t(input, input, continuation, state)

def replace(from_str, to_str, left_ctxt, right_ctxt, input):
    """ Implementation of a replace rule:
    from_str -> to_str || left_ctxt _ right_ctxt

    All strings are regular expressions. Any string can be empty.
    """
    pattern = "(?P<first>" + left_ctxt + ")" + from_str + "(?P<second>" + right_ctxt + ")"
    replacement = r"\g<first>" + to_str + r"\g<second>"

    before = "{:s}".format(input.get_string()) # Make a copy, not a reference
    after = re.sub(pattern, replacement, before)
    while len(before) == len(after) and before != after:
        # Multiple matches with overlapping contexts:
        # not allowed when repeatedly inserting or deleting letters, though
        before = after
        after = re.sub(pattern, replacement, before)

    if input.is_debug():
        print('Rule:    "{:s}" -> "{:s}" || "{:s}" _ "{:s}"'.format(from_str, to_str, left_ctxt, right_ctxt))
        print('Applied: "{:s}" -> "{:s}"'.format(input.get_string(), after))
        print()

    input.set_string(after)

def apply(rules, wordform, debug=False):
    """ Apply the rules to a given wordform. This is for testing.
        If debug is True, then show all intermediate steps
    """
    print("Applying replace rules on", wordform)
    r = ReplacedString(wordform, debug)
    rules(r)
    if debug:
        print("End result:")
    print("  " + wordform, "=>", r.get_string())
    print("")

'''
# This is how things could be done if there were no rules; then we
# would not need to produce all possible word forms in advance and
# apply the rules on the full word forms

def retrieve(input, is_analyzer, return_all):
    """ Retrieve all results and print them """
    state = State("", "", input, is_analyzer, return_all) 
    root(state)
    for (i, o) in state.get_result():
        print(i + " => " + o)
    print()

def analyze(wordform):
    """ Analyze a word form """
    print("Analyze:", wordform)
    retrieve(wordform, True, False)

def generate(analysis):
    """ Generate a word form from an analysis """
    print("Generate:", analysis)
    retrieve(analysis, False, False) 

def analyze_all():
    print("Show all analyses in lexicon")
    retrieve("", True, True)

def generate_all():
    print("Show all word forms in lexicon")
    retrieve("", False, True)
'''

def load_lexicon(root, rules):
    """ The function load_lexicon produces all possible word forms and analyses,
    runs the replace rules, and stores the result in two dicts (= lookup tables)
    """
    wordform2analysis.clear()
    analysis2wordform.clear()
    state = State("", "", "", True, True)
    root(state)
    for (i, o) in state.get_result():
        r = ReplacedString(o)
        if rules is not None:
            rules(r)
        wordform2analysis[r.get_string()].add(i)
        analysis2wordform[i].add(r.get_string())

def analyze(wordform):
    """ Analyze a word form through lookup """
    print("Analyze:", wordform)
    if wordform in wordform2analysis.keys():
        for o in wordform2analysis[wordform]:
            print("  ", o)
    else:
        print("  No matching words found.")
    print()

def generate(analysis):
    """ Generate a word form through lookup """
    print("Generate:", analysis)
    if analysis in analysis2wordform.keys():
        for o in analysis2wordform[analysis]:
            print("  ", o)
    else:
        print("  No matching words found.")
    print()

def analyze_all():
    print("Show all analyses in lexicon")
    for i in wordform2analysis:
        for o in wordform2analysis[i]:
            print(i, " => ", o)
    print()
    
def generate_all():
    print("Show all word forms in lexicon")
    for i in analysis2wordform:
        for o in analysis2wordform[i]:
            print(i, " => ", o)
    print()

def show_random_analyses(n):
    print("Show", n, "random analyses from lexicon")
    for o in random.sample(list(analysis2wordform), n):
        print(o)
    print()

def show_random_wordforms(n):
    print("Show", n, "random word forms from lexicon")
    for o in random.sample(list(wordform2analysis), n):
        print(o)
    print()
