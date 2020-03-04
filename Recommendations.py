import pandas as pd
import numpy as np
from surprise import Reader, Dataset, KNNBasic, BaselineOnly, KNNWithMeans, NMF
from surprise.model_selection import KFold
from collections import defaultdict, Counter
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def mean_of_dictionaries(*args):
    # sum dictionary entries
    counter = Counter()
    for dictionary in args:
        counter += dictionary

    # compute the average
    for key in counter:
        counter[key] /= len(args)

    return dict(counter)

def evaluate_for_each_group(predictions, U1, U2, U3, U4, beyms, ms):
    absolute_errors = defaultdict(list)
    for user_id, item_id, r, r_, details in predictions:
        ae = np.abs(r - r_)
        absolute_errors["all"].append(ae)
        if user_id in beyms:
            absolute_errors["beyms"].append(ae)
        elif user_id in ms:
            absolute_errors["ms"].append(ae)

        if user_id in U1:
            absolute_errors["U1"].append(ae)
        elif user_id in U2:
            absolute_errors["U2"].append(ae)
        elif user_id in U3:
            absolute_errors["U3"].append(ae)
        elif user_id in U4:
            absolute_errors["U4"].append(ae)

    mean_absolute_errors = dict()
    for group in absolute_errors:
        mae = np.mean(absolute_errors[group])
        mean_absolute_errors[group] = mae

    t_statistic, ttest_p = ttest_ind(absolute_errors["ms"], absolute_errors["beyms"])
    ttest_p /= 2
    ttest = {"t_statistic": t_statistic, "p_value": ttest_p}

    f_value, anova_p = f_oneway(absolute_errors["U1"], absolute_errors["U2"], absolute_errors["U3"], absolute_errors["U4"])

    aes_with_user_groups_df = pd.DataFrame()
    aes_with_user_groups_df = aes_with_user_groups_df.append(pd.DataFrame(
        data={"absolute_error": absolute_errors["U1"], "user group": "U1"}))
    aes_with_user_groups_df = aes_with_user_groups_df.append(pd.DataFrame(
        data={"absolute_error": absolute_errors["U2"], "user group": "U2"}))
    aes_with_user_groups_df = aes_with_user_groups_df.append(pd.DataFrame(
        data={"absolute_error": absolute_errors["U3"], "user group": "U3"}))
    aes_with_user_groups_df = aes_with_user_groups_df.append(pd.DataFrame(
        data={"absolute_error": absolute_errors["U4"], "user group": "U4"}))

    tukeyhsd_results = pairwise_tukeyhsd(endog=aes_with_user_groups_df["absolute_error"],
                                         groups=aes_with_user_groups_df["user group"],
                                         alpha=0.05)
    pairwise_comparison = {"ANOVA": {"f_value": f_value, "p_value": anova_p}, "TukeyHSD": tukeyhsd_results.summary()}

    return mean_absolute_errors, ttest, pairwise_comparison



if __name__ == "__main__":
    # load necessary datasets
    user_groups_df = pd.read_csv("data/user_groups.csv")
    U1 = user_groups_df[user_groups_df["user group"] == 0]["user_id"].tolist()
    U2 = user_groups_df[user_groups_df["user group"] == 1]["user_id"].tolist()
    U3 = user_groups_df[user_groups_df["user group"] == 2]["user_id"].tolist()
    U4 = user_groups_df[user_groups_df["user group"] == 3]["user_id"].tolist()
    beyms = pd.read_csv("data/beyms.csv")["user_id"].tolist()
    ms = pd.read_csv("data/ms.csv")["user_id"].tolist()

    ratings_df = pd.read_csv("data/ratings.csv")

    # compute recommendations
    UserItemAvg_errors_per_fold, UserKNN_errors_per_fold, UserKNNAvg_errors_per_fold, NMF_errors_per_fold = [], [], [], []

    reader = Reader(rating_scale=(0, 1000))
    dataset = Dataset.load_from_df(ratings_df, reader)
    folds_it = KFold(n_splits=5).split(dataset)
    for f, data in enumerate(folds_it):
        trainset, testset = data

        """print("==========================================================================")
        print("[Fold %d], UserItemAvg" % f)
        print("==========================================================================")
        UserItemAvg_preds = BaselineOnly().fit(trainset).test(testset)
        errors, ttest_results, pairwise_results = evaluate_for_each_group(UserItemAvg_preds, U1, U2, U3, U4, beyms, ms)
        UserItemAvg_errors_per_fold.append(errors)
        print("Mean Absolute Error: " + str(errors))
        print("t-Test: " + str(ttest_results))
        print("ANOVA: " + str(pairwise_results["ANOVA"]))
        print(pairwise_results["TukeyHSD"])

        print("==========================================================================")
        print("[Fold %d], UserKNN" % f)
        print("==========================================================================")
        UserKNN_preds = KNNBasic(k=40, sim_options={"name": "cosine"}).fit(trainset).test(testset)
        errors, ttest_results, pairwise_results = evaluate_for_each_group(UserKNN_preds, U1, U2, U3, U4, beyms, ms)
        UserKNN_errors_per_fold.append(errors)
        print("Mean Absolute Error: " + str(errors))
        print("t-Test: " + str(ttest_results))
        print("ANOVA: " + str(pairwise_results["ANOVA"]))
        print(pairwise_results["TukeyHSD"])

        print("==========================================================================")
        print("[Fold %d], UserKNNAvg" % f)
        print("==========================================================================")
        UserKNNAvg_preds = KNNWithMeans(k=40, sim_options={"name": "cosine"}).fit(trainset).test(testset)
        errors, ttest_results, pairwise_results = evaluate_for_each_group(UserKNNAvg_preds, U1, U2, U3, U4, beyms, ms)
        UserKNNAvg_errors_per_fold.append(errors)
        print("Mean Absolute Error: " + str(errors))
        print("t-Test: " + str(ttest_results))
        print("ANOVA: " + str(pairwise_results["ANOVA"]))
        print(pairwise_results["TukeyHSD"])"""

        print("==========================================================================")
        print("[Fold %d], NMF" % f)
        print("==========================================================================")
        NMF_preds = NMF(n_factors=15).fit(trainset).test(testset)
        errors, ttest_results, pairwise_results = evaluate_for_each_group(NMF_preds, U1, U2, U3, U4, beyms, ms)
        NMF_errors_per_fold.append(errors)
        print("Mean Absolute Error: " + str(errors))
        print("t-Test: " + str(ttest_results))
        print("ANOVA: " + str(pairwise_results["ANOVA"]))
        print(pairwise_results["TukeyHSD"])


    print("[MAE over all folds] UserItemAvg: " + str(mean_of_dictionaries(*UserItemAvg_errors_per_fold)))
    print("[MAE over all folds] UserKNN: " + str(mean_of_dictionaries(*UserKNN_errors_per_fold)))
    print("[MAE over all folds] UserKNNAvg: " + str(mean_of_dictionaries(*UserKNNAvg_errors_per_fold)))
    print("[MAE over all folds] NMF: " + str(mean_of_dictionaries(*NMF_errors_per_fold)))







