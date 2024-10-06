"""
PHYS20161 Assignment 2: 79_Rb decay

Utility module for supporting the decay programme.

@author: e94928lv
Last Updated: 13/12/22
"""

import os
import time
import sys

PER_SECOND = "s" + "\u207B" + "\u00B9"
PLUS_MINUS = "\u00B1"
CHI_SQUARED = "\u2CAC" + "\u00B2"
DECAY_CONSTANT_RB = "\u03BB" + "-RB"
DECAY_CONSTANT_SR = "\u03BB" + "-SR"
HALF_LIFE = "t\u00BD"


def cls():
    """
    Clears the console before starting the program.
    Checks first if we are running on Windows or Linux
    """
    os.system('cls' if os.name == 'nt' else 'clear')


Style = {'BLACK': '\033[30m', 'RED': '\033[31m', 'GREEN': '\033[32m',
         'YELLOW': '\033[33m', 'BLUE': '\033[34m', 'MAGENTA': '\033[35m',
         'CYAN': '\033[36m', 'WHITE': '\033[37m', 'UNDERLINE': '\033[4m',
         'RESET': '\033[0m'}


def print_title():
    """
    Prints the title message for the programme
    """
    print(Style['GREEN'] + "\n************************************************"
          + Style['RESET'])
    print(Style['YELLOW'] + " PHYS20161 2nd assignment: 79Rb Decay")
    print(" December 13th, 2022")
    print(" 10828724")
    print(Style['GREEN'] + "************************************************\n"
          + Style['RESET'])


def get_int_input(message):
    """
    finds  the number of files the user would like to input and ensures this
    number is an integer
    :param message: string
    :return: value: integer
    """
    while True:
        value = input(message)
        try:
            value = int(value)
            if value > 0:
                return value
            else:
                sys.exit()
        except ValueError:
            print('The number must be an integer')


def validate_positive_input(value):
    """
    Validate positive value. Assumes value is a float.
    :param value: float
    :return: True or False
    """
    if float(value) <= 0.0:
        print(Style['RED'] + "ERROR: " + Style['RESET']
              + "Input must be a positive value! Please try again.")
        return False
    return True


def validate_y_n_input(question):
    """
    Ensures the questions are answered correctly
    :param question: string
    :return: answer: string
    """
    while True:
        answer = input(question).lower()
        if answer not in ('y', 'n'):
            print(Style['RED'] + "ERROR: " + Style['RESET']
                  + "Input must be y or n.")
        else:
            return answer


def print_cancelled_by_user():
    """
    Prints message when user cancels programme.
    """
    print(Style['RED'] + "\nCancelled by user." + Style['RESET'])


def print_explanation():
    """
    Prints the presentation/explanation of the programme
    """
    print('This program reads in data files that include a time in hours, an '
          'activity reading in TBq (tera-Bequerels), and the uncertainty on '
          'that activity reading.\nFrom this data, it calculates a best-fit '
          'estimate of the decay constant and then the half-life of the two '
          'nuclides with uncertainties.\nThe programme also calculates the '
          'reduced chi^2 and has a few extra features.\n')


def benchmark(func):
    """
    Benchmark decorator to calculate the total time of a func
    :param func: function
    """
    def benchmark_func(*args, **kwargs):
        """
        Benchmark function replacing func.
        :param args: list of arguments from original function
        :param kwargs: list of key-value pairs form original function
        """
        time_1 = time.time()
        result = func(*args, **kwargs)
        time_2 = time.time()
        print(Style['RED'] + "Benchmark: " + Style['RESET'] + Style['YELLOW']
              + "Function: %s, took: %.2f seconds to finish processing" %
              (func.__name__, time_2 - time_1)
              + Style['RESET'])
        return result

    return benchmark_func
