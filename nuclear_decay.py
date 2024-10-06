"""
===============================================================================
PHYS20161 Assignment 2: 79_Rb decay

This program reads in data files that include a time in hours, an activity
reading in TBq (tera-Bequerels), and the uncertainty on that activity
reading. From this data, it calculates a best-fit estimate of the decay
constant and then the half-life of the two nuclides with uncertainties.
The programme also calculates the reduced chi^2 and has a few extra features.

@author: e94928lv
Last Updated: 13/12/22
===============================================================================
"""

import os
from collections import namedtuple
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import decay_utils

SAMPLE_SIZE = 0.000001 * 6.022 * 10 ** 23
SR_DECAY_CONSTANT = 0.005
RB_DECAY_CONSTANT = 0.0005

TRIAL_SOLUTIONS = (RB_DECAY_CONSTANT, SR_DECAY_CONSTANT)

Results = namedtuple("Results",
                     "lambda_rb uncertainty_half_life_rb lambda_rb_uncertainty"
                     " half_life_rb lambda_sr uncertainty_half_life_sr "
                     "lambda_sr_uncertainty half_life_sr reduced_chi_squared "
                     "uncertainty_activity_tbq iterations function_calls")


def activity_function(time, lambda_rb, lambda_sr):
    """
    Calculates the expected activity based on the given formula.
    :param time: floats
    :param lambda_rb: float
    :param lambda_sr: float
    :return: activity: float
    """
    activity = SAMPLE_SIZE * ((lambda_sr * lambda_rb) / (lambda_rb - lambda_sr)
                              ) * (np.exp((-lambda_sr) * time) -
                                   np.exp((-lambda_rb) * time)) / 10 ** 12
    return activity


def get_filenames():
    """
    Asks the user what data files they would like to use and returns a list
    with the names of the files to be used once they have been validated. If
    the default data files the programme will quite and raise a warning to the
    user.
    :return: data: list[strings]
    """
    answer = decay_utils.validate_y_n_input(
        "Would you like to use the default data files (these should be located"
        " in the same directory as this programme)? (y/n) ")

    file_list = ['Nuclear_data_1.csv', 'Nuclear_data_2.csv']

    new_data = answer == 'n'

    if not new_data:
        if not os.path.exists(file_list[0]) or not os.path.exists(file_list[1]
                                                                  ):
            print(decay_utils.Style['RED'] + '\nERROR: '
                  + 'Cannot find default files \'Nuclear_data_1.csv\' and \'Nu'
                    'clear_data_2.csv\'. '
                  + 'Make sure they are in the same directory as this program!'
                  + decay_utils.Style['RESET'])
            sys.exit()

    if new_data:
        file_list = []
        number_of_files = (decay_utils.get_int_input
                           ('How many files would you like to enter: '))
        for i in range(number_of_files):
            while True:
                temp = input(
                    'Please enter the full path to the file you would like to '
                    'add: ')
                if os.path.exists(temp):
                    if temp.endswith('.csv'):
                        file_list.append(temp)
                        break
                    else:
                        print(decay_utils.Style['RED'] + 'The file must be a '
                                                         'csv file.' +
                              decay_utils.Style['RESET'])
                else:
                    print(decay_utils.Style['RED'] + 'The file cannot be found'
                                                     ', try again.' +
                          decay_utils.Style['RESET'])
    return file_list


def combine_files(file_list):
    """
    Combines all the files into a single numpy array and removes the nan's.
    :param file_list: list[strings]
    :return: filtered_data_1: array[floats]
    """
    combined_data = np.genfromtxt(file_list[0], delimiter=',', comments='%',
                                  skip_header=1)
    for i in range(len(file_list) - 1):
        temp = np.genfromtxt(file_list[i + 1], delimiter=',', comments='%',
                             skip_header=1)
        combined_data = np.vstack((combined_data, temp))
    combined_data = combined_data[~np.isnan(combined_data).any(axis=1)]
    return combined_data


def filter_data(combined_data, trial_solutions):
    """
    Filters all the data by removing outliers.
    :param combined_data: array[floats]
    :param trial_solutions: array[floats]
    :return: filtered_data: array[floats]
    """
    index_array = np.array([])
    for count, time in enumerate(combined_data[:, 0]):
        if abs(activity_function(time * 60 ** 2, trial_solutions[0],
                                 trial_solutions[1]) - combined_data[
                   count, 1]) > 3 * combined_data[count, 2]:
            index_array = np.append(index_array, count)
    index_array = np.array(index_array, int)
    filtered_data = np.delete(combined_data, index_array, axis=0)
    return filtered_data


def get_expected_data(combined_data, trial_solutions):
    """
    Gets an array of expected activity using the activity_function.
    :param combined_data: array[floats]
    :param trial_solutions: array[floats]
    :return: expected_data: array[floats]
    """
    maximum = np.amax(combined_data[:, 0] * 60 ** 2)
    minimum = np.min(combined_data[:, 0] * 60 ** 2)
    times = np.linspace(minimum, maximum, 1000)
    expected_activity_data = np.array([])
    for time in times:
        function = activity_function(time, trial_solutions[0],
                                     trial_solutions[1])
        expected_activity_data = np.append(expected_activity_data, function)

    expected_data = np.vstack([times, expected_activity_data]).T

    return expected_data


def calculate_chi_squared(lambda_array, time, activity, uncertainty_data):
    """
    Calculates the chi^2 value given the measured data.
    :param lambda_array: array[floats]
    :param time: array[floats]
    :param activity: array[floats]
    :param uncertainty_data: array[floats]
    :return: chi_squared: float
    """
    lambda_rb = lambda_array[0]
    lambda_sr = lambda_array[1]

    chi_squared = np.sum((activity - activity_function(time * 60 ** 2,
                                                       lambda_rb, lambda_sr))
                         ** 2 / uncertainty_data ** 2)

    return chi_squared


def build_mesh(lambda_rb, lambda_sr, data):
    """
    Create the meshes for the contour plot.
    :param lambda_rb: float
    :param lambda_sr: float
    :param data: array[floats]
    :return: x_axis_mesh: array[floats]
    :return: y_axis_mesh: array[floats]
    :return: chi_value_mesh: array[floats]
    """
    multiplier = 0.95
    x_axis = np.linspace(lambda_rb * multiplier, lambda_rb * (1 / multiplier),
                         len(data[:, 0]))
    y_axis = np.linspace(lambda_sr * multiplier, lambda_sr * (1 / multiplier),
                         len(data[:, 0]))

    chi_value_mesh = np.empty(len(x_axis) * len(y_axis))
    x_axis_mesh, y_axis_mesh = np.meshgrid(x_axis, y_axis)
    chi_value_mesh = chi_value_mesh.reshape(x_axis_mesh.shape)
    for row in range(0, len(data[:, 0])):
        for column in range(0, len(data[:, 0])):
            chi_value_mesh[row][column] = calculate_chi_squared(
                (x_axis_mesh[row][column], y_axis_mesh[row][column]),
                data[:, 0], data[:, 1], data[:, 2])
    return x_axis_mesh, y_axis_mesh, chi_value_mesh


def plot_graphs(lambda_rb, lambda_sr, min_chi_squared, data, expected_data):
    """
    Produces a graph of activity vs time and a contour plot on the same figure.
    :param lambda_rb: float
    :param lambda_sr: float
    :param min_chi_squared: float
    :param data: array[floats]
    :param expected_data: array[floats]
    :return: rb_uncertainty : float
    :return: sr_uncertainty : float
    """
    fig = plt.figure(figsize=(13, 5))

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_title('Activity vs Time.', fontsize=16, color='black')
    ax1.set_xlabel('Time(s)', fontsize=12, color='black')
    ax1.set_ylabel('Activity(TBq)', fontsize=12, color='black')

    ax1.tick_params(labelsize=14)

    ax1.errorbar(data[:, 0] * 60 ** 2, data[:, 1],
                 yerr=data[:, 2], fmt='.', label='Data')
    ax1.plot(expected_data[:, 0], expected_data[:, 1], label='Expected',
             color='red')
    ax1.tick_params(labelsize=14)
    ax1.legend(loc='upper right', fontsize=14)

    get_mesh = build_mesh(lambda_rb, lambda_sr, data)

    ax2.set_title(
        r'$\chi^2$ contours against $\lambda_{Rb}$ and $\lambda_{Sr}$')
    ax2.set_xlabel(r'$\lambda_{Rb}$')
    ax2.set_ylabel(r'$\lambda_{Sr}$')
    ax2.ticklabel_format(scilimits=(0, 0))
    ax2.scatter(lambda_rb, lambda_sr, color='w', marker='x',
                label=r'$\chi^2_{\min}$ = 'f'{min_chi_squared:.1f}')
    ax2.text(lambda_rb * 1.009, lambda_sr * 0.999,
             f'{lambda_rb:.2E}, {lambda_sr:.2E}', color='black')
    contour_plot = ax2.contourf(get_mesh[0], get_mesh[1], get_mesh[2], 14,
                                zorder=0)
    ellipse = ax2.contour(get_mesh[0], get_mesh[1], get_mesh[2], 1,
                          levels=[min_chi_squared + 1],
                          linestyles='dashdot', colors='w')
    std_array = ellipse.allsegs[0][0]
    rb_uncertainty = np.max(std_array[:, 0] - np.min(std_array[:, 0])) / 2
    sr_uncertainty = np.max(std_array[:, 1] - np.min(std_array[:, 1])) / 2

    ax2.clabel(ellipse, colors='k')

    plt.legend()
    fig.colorbar(contour_plot)
    plt.savefig('decay_plots', dpi=600)
    plt.ion()
    plt.show()
    plt.pause(0.001)
    return rb_uncertainty, sr_uncertainty


def activity_uncertainty(time, lambda_rb, lambda_sr, rb_uncertainty,
                         sr_uncertainty):
    """
    Uses error propagation to calculate the uncertainty on the activity.
    :param time: floats
    :param lambda_rb: float
    :param lambda_sr: float
    :param rb_uncertainty: float
    :param sr_uncertainty: float
    :return: uncertainty_activity_tbq: float
    """
    da_drb = SAMPLE_SIZE * (lambda_sr * np.exp(time * (- lambda_rb - lambda_sr)
                                               ) * (np.exp(lambda_sr * time) *
                                                    ((lambda_rb ** 2) * time -
                                                     lambda_rb * lambda_sr *
                                                     time + lambda_sr) -
                                                    lambda_sr *
                                                    np.exp(lambda_rb * time))
                            ) / (lambda_rb - lambda_sr) ** 2
    da_dsr = - SAMPLE_SIZE * lambda_rb * np.exp(time * (- lambda_rb -
                                                        lambda_sr)) * (
            np.exp(lambda_rb * time) * (lambda_rb * (lambda_sr * time - 1) -
                                        (lambda_sr ** 2) * time) + lambda_rb
            * np.exp(lambda_sr * time)) / (lambda_rb - lambda_sr) ** 2
    uncertainty_activity = np.sqrt((da_drb * rb_uncertainty) ** 2 +
                                   (da_dsr * sr_uncertainty) ** 2)
    uncertainty_activity_tbq = uncertainty_activity / (10 ** 12)
    return uncertainty_activity_tbq


def print_results(results: Results):
    """
    Prints all the results.
    :param results: namedtuple
    """
    print(
        f'\nThe Rb decay constant ({decay_utils.DECAY_CONSTANT_RB}) is: '
        f'{results.lambda_rb:#.3} {decay_utils.PLUS_MINUS} '
        f'{results.lambda_rb_uncertainty:#.1} {decay_utils.PER_SECOND}')
    print(
        f'The Sr decay constant ({decay_utils.DECAY_CONSTANT_SR}) is: '
        f'{results.lambda_sr:#.3} {decay_utils.PLUS_MINUS} '
        f'{results.lambda_sr_uncertainty:#.2} {decay_utils.PER_SECOND}\n')
    print(
        f'Rb half-life ({decay_utils.HALF_LIFE}) is: '
        f'{results.half_life_rb:#.3} {decay_utils.PLUS_MINUS} '
        f'{results.uncertainty_half_life_rb:#.1} minutes')
    print(
        f'Sr half-life ({decay_utils.HALF_LIFE}) is: {results.half_life_sr:.3}'
        f' {decay_utils.PLUS_MINUS} '
        f'{results.uncertainty_half_life_sr:#.1} minutes\n')
    print('The number of iterations this took was: ', results.iterations)
    print('The number of function calls were: ', results.function_calls, '\n')
    print(
        f'The reduced chi-squared ({decay_utils.CHI_SQUARED}) is:'
        f' {results.reduced_chi_squared:#.2f}\n')
    print(f'The predicted activity level (in TBq) at t = 90 minutes is: 'f''
          f'{activity_function(90 * 60, results.lambda_rb, results.lambda_sr):#.3} {decay_utils.PLUS_MINUS} {results.uncertainty_activity_tbq:#.1} \n')


@decay_utils.benchmark
def main_calculation(file_names):
    """
    Entry point for programme.
    :param file_names: array[strings]
    """
    total_data = combine_files(file_names)
    calculated_data = get_expected_data(total_data, TRIAL_SOLUTIONS)

    final_data = filter_data(total_data, TRIAL_SOLUTIONS)

    measured_time = final_data[:, 0]
    activity = final_data[:, 1]
    activity_uncertainty_data = final_data[:, 2]

    minimised_function = fmin(calculate_chi_squared, TRIAL_SOLUTIONS,
                              args=(measured_time, activity,
                                    activity_uncertainty_data),
                              full_output=True, disp=False)

    decay_uncertainties = plot_graphs(minimised_function[0][0],
                                      minimised_function[0][1],
                                      minimised_function[1], final_data,
                                      calculated_data)

    rb_lambda = minimised_function[0][0]
    sr_lambda = minimised_function[0][1]

    iterations = minimised_function[2]
    function_calls = minimised_function[3]

    half_life_rb = np.log(2) / rb_lambda / 60
    half_life_sr = np.log(2) / sr_lambda / 60

    lambda_rb_uncertainty = decay_uncertainties[0]
    lambda_sr_uncertainty = decay_uncertainties[1]

    fractional_uncertainty_rb = lambda_rb_uncertainty / TRIAL_SOLUTIONS[0]
    fractional_uncertainty_sr = lambda_sr_uncertainty / TRIAL_SOLUTIONS[1]

    uncertainty_half_life_rb = fractional_uncertainty_rb * half_life_rb
    uncertainty_half_life_sr = fractional_uncertainty_sr * half_life_sr

    reduced_chi_squared = minimised_function[1] / (len(final_data) - 2)

    uncertainty_activity_tbq = activity_uncertainty(90 * 60, rb_lambda,
                                                    sr_lambda,
                                                    lambda_rb_uncertainty,
                                                    lambda_sr_uncertainty)

    results = Results(rb_lambda, uncertainty_half_life_rb,
                      lambda_rb_uncertainty, half_life_rb,
                      sr_lambda, uncertainty_half_life_sr,
                      lambda_sr_uncertainty, half_life_sr,
                      reduced_chi_squared, uncertainty_activity_tbq,
                      iterations, function_calls)
    print_results(results)


def loop():
    """
    Keep looping until user quits.
    """
    new_calc = True
    while new_calc:
        file_names = get_filenames()
        main_calculation(file_names)

        answer = decay_utils.validate_y_n_input(decay_utils.Style['GREEN'] +
                                                "\nDo you want to run the code"
                                                " again? (y/n) " +
                                                decay_utils.Style['RESET'])
        new_calc = answer == "y"
        plt.close()

    print("Bye!")


if __name__ == "__main__":
    try:
        #  decay_utils.cls()
        decay_utils.print_title()
        decay_utils.print_explanation()
        loop()

    except KeyboardInterrupt:
        decay_utils.print_cancelled_by_user()
