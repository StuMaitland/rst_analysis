import pandas as pd
import numpy as np
import plotly.express as px
import sys
import getopt


# csv_loc = '/Users/stuartbman/GitHub/rst_analysis/data/RJ/2021-05-27 13_36_32.817525.txt'


def main(argv):
    inputfile = False
    outputfile = False
    help_str = 'force_corr.py -i <inputfile> -o <outputfile>'

    try:
        opts, args = getopt.getopt(argv, 'hi:o:', ['ifile=', 'ofile='])
    except getopt.GetoptError:
        print(help_str)
        sys.exit()
    for opt, arg in opts:
        if opt == '-h':
            print(help_str)
        elif opt in ('-i', '--ifile'):
            inputfile = arg
        elif opt in ('-o', '--ofile'):
            outputfile = arg

    if inputfile:
        df = pd.read_csv(inputfile,
                         skiprows=2,
                         names=['target_digit', 'target_force', 't', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5']
                         )
    else:
        raise FileNotFoundError(help_str)

    if not outputfile:
        outputfile = inputfile[:-4]

    mvcs = pd.read_csv(inputfile, header=None, skiprows=1, nrows=1)
    for n in range(1, 6):
        df['sf_{}'.format(n)] = df['f_{}'.format(n)] / mvcs[n - 1][0]

    df['at_rest'] = np.where(df['target_digit'] == -1, 1, 0)

    for n in range(1, 6):
        df['tf_{}'.format(n)] = np.where(df['target_digit'] == n - 1, df['target_force'] / mvcs[n - 1][0], 0)

    df = df.assign(new=df.target_digit.diff().ne(0).cumsum())

    result_df = pd.DataFrame(
        columns=['group_id', 'target_digit', 'target_force', 'max_1', 'max_2', 'max_3', 'max_4', 'max_5'])

    for n in df.new.unique():
        target_digit = df[df['new'] == n]['target_digit'].iloc[0] + 1
        if target_digit > 0:
            row = []
            row.append(n)  # group index

            row.append(target_digit)  # target digit
            row.append(
                df[df['new'] == n]['tf_{}'.format(target_digit)].iloc[0] / mvcs[target_digit - 1][0])  # target force

            for i in range(1, 6):
                max_f = df[df['new'] == n]['sf_{}'.format(i)].mean()
                row.append(max_f)
            result_df.loc[len(result_df)] = row

    digcorr = np.zeros((5, 5))
    for dig in range(1, 6):
        digit = result_df[result_df['target_digit'] == dig]
        for dig2 in range(1, 6):
            digcorr[dig - 1, dig2 - 1] = np.corrcoef(digit['target_force'], digit['max_{}'.format(dig2)])[0][1]

    if outputfile:
        np.savetxt("{}_results.txt".format(outputfile), digcorr, delimiter=",", fmt='%1.3f')
        fig = px.line(df, x='t', y=['sf_1', 'sf_2', 'sf_3', 'sf_4', 'sf_5', 'tf_1', 'tf_2', 'tf_3', 'tf_4', 'tf_5'])
        fig.write_image("{}_force_graph.png".format(outputfile))
    else:
        print(digcorr)
    # step 0- standardise all forces to MVC
    # step 1- simple relationship between target force & target strength (simple R^2)
    # step 2- measure force leakage- perhaps a simple R2 between target strength & mean force on other digits

    # fig = px.scatter(digit, x='target_force', y=['max_1', 'max_2', 'max_3', 'max_4', 'max_5'])

    # fig.show()


# fig.show()

if __name__ == '__main__':
    main(sys.argv[1:])
