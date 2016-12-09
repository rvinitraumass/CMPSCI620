from __future__ import print_function
from __future__ import division
import sys
import pandas as pd
import numpy as np

np.random.seed(42)


def read_data(csvname):
    '''
    Read the input CSV file
    '''
    data = pd.read_csv(csvname)
    # Add a method column
    data["Method"] = data.Family + "/" + data.Formula
    return data


def get_means(data, groupby=["Method"]):
    return data.groupby(groupby).mean()


def get_stds(data, groupby=["Method"]):
    return data.groupby(groupby).std()


def get_artificial_real_means(data, groupby=["Method"]):
    '''
    Given dataframe and column to "groupby"
    return two dataframes for artificial and real bugs
    containing the mean of each column grouped by each category in groupby
    '''
    artificial = data.loc[data["Bug"] > 1000]
    real = data.loc[data["Bug"] < 1000]
    arti_means = artificial.groupby(groupby).mean()
    real_means = real.groupby(groupby).mean()
    return (arti_means, real_means)


def add_folds_column(data, numfolds=5):
    folds = np.random.choice(
        np.arange(numfolds), size=data.shape[0], replace=True
    )
    data["Fold"] = folds
    return data


# def fold_means_artificial_real(data, numfolds=5):
#     fold_data = add_folds_column(data, numfolds)
#     return get_artificial_real_means(fold_data, ["Method", "Fold"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <csv-file>")
        sys.exit()

    fname = sys.argv[1].strip()
    data = read_data(fname)
    print(data.head())
    art_means, real_means = get_artificial_real_means(data)
    exam = "ScoreWRTLoadedClasses"
    print("-"*50)
    print("Artificial Bugs Scores:")
    print(art_means[exam])
    print("-"*50)
    print("Real Bugs Scores:")
    print(real_means[exam])

    # average scores from cross-validation
    numfolds = 5
    fold_data = add_folds_column(data, numfolds)
    (art_fold_means, real_fold_means) = get_artificial_real_means(fold_data, ["Method", "Fold"])
    print("-"*50)
    print("Artificial Bugs Scores for all folds:")
    print(art_fold_means[exam])
    print("-"*50)
    print("Real Bugs Scores for all folds:")
    print(real_fold_means[exam])

    print("EXAM score analysis from %d-fold cross validation" % (numfolds))
    art_fold_means.reset_index(inplace=True)
    real_fold_means.reset_index(inplace=True)
    print(art_fold_means.head())
    print("-"*50)
    print("Mean Artificial Bugs Scores:")
    print(get_means(art_fold_means[["Method", exam]]))
    print("Standard Deviation of Artificial Bugs Scores:")
    print(get_stds(art_fold_means[["Method", exam]]))
    print("")
    print("-"*50)
    print("Mean Real Bugs Scores:")
    print(get_means(real_fold_means[["Method", exam]]))
    print("Standard Deviation of Real Bugs Scores:")
    print(get_stds(real_fold_means[["Method", exam]]))

    print("\n")
    print("EXAM score analysis for Each project Group from %d-fold cross validation" % (numfolds))
    # print(fold_data.head())
    (art_fold_means, real_fold_means) = get_artificial_real_means(fold_data, ["Project", "Method", "Fold"])
    art_fold_means.reset_index(inplace=True)
    real_fold_means.reset_index(inplace=True)
    print("-"*50)
    print("Mean Artificial Bugs Scores:")
    print(get_means(art_fold_means[["Method", "Project", exam]], groupby=["Project", "Method"]))
    print("Standard Deviation of Artificial Bugs Scores:")
    print(get_stds(art_fold_means[["Method", "Project", exam]], groupby=["Project", "Method"]))
    print("")
    print("-"*50)
    print("Mean Real Bugs Scores:")
    print(get_means(real_fold_means[["Method", "Project", exam]], groupby=["Project", "Method"]))
    print("Standard Deviation of Real Bugs Scores:")
    print(get_stds(real_fold_means[["Method", "Project", exam]], groupby=["Project", "Method"]))


    # Analyze the mean/std between performance on artifical and real faults
    artbug_data = fold_data.loc[fold_data.Bug > 1000]
    artbug_data.loc[:,"RBugId"] = np.floor(artbug_data.Bug / 100000)
    # Mean scores for each fold corresponding to each real Bug
    artbug_data.reset_index(inplace=True)
    artbug_data_means = get_means(artbug_data, ["Project", "Method", "Fold", "RBugId"])
    artbug_data_means = artbug_data_means.loc[:, [exam]]
    artbug_data_means.index = artbug_data_means.index.set_names("Bug", level=3)
    artbug_data_means.reset_index(inplace=True)
    # print(artbug_data_means.head())
    # artbug_data_means.reset_index(inplace=True)
    # print(artbug_data_means)

    realbug_data = fold_data.loc[fold_data.Bug < 1000]
    # Mean scores for each fold corresponding to each real Bug
    # realbug_data.reset_index(inplace=True)
    realbug_data = realbug_data.loc[:, ["Project", "Method", "Fold", "Bug", exam]] #.groupby(["Project", "Method", "Fold", "Bug"])
    # realbug_data = realbug_data.set_index(["Project", "Method", "Fold", "Bug"])
    # print(realbug_data.head())
    # print(artbug_data_means.head())
    means_real_art_data = realbug_data.merge(
        artbug_data_means, left_index=False, right_index=False,
        on=["Project", "Method", "Fold", "Bug"],
        suffixes=("_real", "_art")
    )
    # RESULTS
    # NOTE: Not splitting by Project as that leads to
    # very little data in each cut and thus very high variance
    # Mean scores for each fold
    mean_fold_scores = means_real_art_data.groupby(["Method", "Fold"]).mean()
    mean_fold_scores.reset_index(inplace=True)
    # At this point we have the scores for artificial and real bugs for each fold
    mean_scores = mean_fold_scores.groupby(["Method"]).mean()
    print("Mean Scores for real and artifical bugs from %d folds:" % numfolds)
    print(mean_scores[[exam+"_real", exam+"_art"]])
    std_scores = mean_fold_scores.groupby(["Method"]).std()
    print("Std of Scores for real and artifical bugs from %d folds:" % numfolds)
    print(std_scores[[exam+"_real", exam+"_art"]])

    # Correlation between real and artifical
    print(mean_fold_scores.head())
    # means_real_art_data[["Project", "Method", ""]]
    print("*"*50)
    print("Correlation Matrix")
    print(mean_fold_scores[[exam+"_real", exam+"_art"]].corr())
