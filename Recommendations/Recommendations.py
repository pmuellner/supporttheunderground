import pandas as pd
import numpy as np
from surprise import Reader, Dataset, KNNBasic, BaselineOnly, KNNWithMeans, NMF, NormalPredictor
from surprise.model_selection import KFold
from collections import defaultdict, Counter
from scipy.stats import ttest_ind, f_oneway, shapiro, mannwhitneyu, norm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import pylab
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

def mean_of_dictionaries(*args):
    # sum dictionary entries
    counter = Counter()
    for dictionary in args:
        counter += Counter(dictionary)

    # compute the average
    for key in counter:
        counter[key] /= len(args)

    return dict(counter)

def evaluate_for_each_group(predictions, U1, U2, U3, U4, beyms, ms):
    list_of_error_tuples = []
    absolute_errors = defaultdict(list)
    for user_id, item_id, r, r_, details in predictions:
        ae = np.abs(r - r_)
        absolute_errors["all"].append(ae)
        list_of_error_tuples.append((user_id, "all", ae))
        if user_id in beyms:
            absolute_errors["beyms"].append(ae)
            list_of_error_tuples.append((user_id, "beyms", ae))
        elif user_id in ms:
            absolute_errors["ms"].append(ae)
            list_of_error_tuples.append((user_id, "ms", ae))
        if user_id in U1:
            absolute_errors["U1"].append(ae)
            list_of_error_tuples.append((user_id, "U1", ae))
        elif user_id in U2:
            absolute_errors["U2"].append(ae)
            list_of_error_tuples.append((user_id, "U2", ae))
        elif user_id in U3:
            absolute_errors["U3"].append(ae)
            list_of_error_tuples.append((user_id, "U3", ae))
        elif user_id in U4:
            absolute_errors["U4"].append(ae)
            list_of_error_tuples.append((user_id, "U4", ae))

    df = pd.DataFrame(list_of_error_tuples)
    df.columns = ["user_id", "group", "mae"]
    mae_per_user_df = df.groupby(["group", "user_id"])["mae"].mean().to_frame()

    mean_absolute_errors = dict()
    for group in absolute_errors:
        mae = np.mean(absolute_errors[group])
        mean_absolute_errors[group] = mae

    absolute_errors["all"] = mae_per_user_df.loc["all"]["mae"].values.tolist()
    absolute_errors["beyms"] = mae_per_user_df.loc["beyms"]["mae"].values.tolist()
    absolute_errors["ms"] = mae_per_user_df.loc["ms"]["mae"].values.tolist()
    absolute_errors["U1"] = mae_per_user_df.loc["U1"]["mae"].values.tolist()
    absolute_errors["U2"] = mae_per_user_df.loc["U2"]["mae"].values.tolist()
    absolute_errors["U3"] = mae_per_user_df.loc["U3"]["mae"].values.tolist()
    absolute_errors["U4"] = mae_per_user_df.loc["U4"]["mae"].values.tolist()

    shapiro_wilk = {"MS": shapiro(absolute_errors["ms"]), "BeyMS": shapiro(absolute_errors["beyms"])}
    print(shapiro_wilk)

    u_statistic, p = mannwhitneyu(absolute_errors["ms"], absolute_errors["beyms"], alternative="less")
    z = norm.ppf(p)
    n = len(ms) + len(beyms)
    r = z / np.sqrt(n)
    mannwhitneyutest = {"MS >= BeyMS": {"U": u_statistic, "p_value": p, "r": r}}

    u_statistic, p = mannwhitneyu(absolute_errors["U3"], absolute_errors["ms"], alternative="less")
    z = norm.ppf(p)
    n = len(U3) + len(ms)
    r = z / np.sqrt(n)
    mannwhitneyutest["U3 >= MS"] = {"U3 >= MS": {"U": u_statistic, "p_value": p, "r": r}}

    t_statistic, ttest_p = ttest_ind(absolute_errors["ms"], absolute_errors["beyms"])
    ttest_p /= 2
    ttest = {"MS >= BeyMS": {"t_statistic": t_statistic, "p_value": ttest_p}}

    t_statistic, ttest_p = ttest_ind(absolute_errors["U3"], absolute_errors["ms"])
    ttest_p /= 2
    ttest["U3 >= MS"] = {"t_statistic": t_statistic, "p_value": ttest_p}

    f_value, anova_p = f_oneway(absolute_errors["U1"], absolute_errors["U2"], absolute_errors["U3"], absolute_errors["U4"])

    df = mae_per_user_df.reset_index()
    df = df[df["group"].isin(["U1", "U2", "U3", "U4"])]
    grand_mean = df["mae"].mean()
    group_mean = df.groupby("group")["mae"].mean()
    ss_total = np.sum(np.power(df["mae"] - grand_mean, 2))
    df = group_mean.to_frame().merge(df, left_index=True, right_on="group")
    df.columns = ["group_mean", "group", "user_id", "user_mae"]
    ss_between = np.sum(np.power(df["group_mean"] - grand_mean, 2))
    eta_squared = ss_between / ss_total

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
    pairwise_comparison = {"ANOVA": {"f_value": f_value, "p_value": anova_p, "eta_squared": eta_squared}, "TukeyHSD": tukeyhsd_results.summary()}

    return mean_absolute_errors, ttest, pairwise_comparison, mannwhitneyutest



if __name__ == "__main__":
    # load necessary datasets
    user_groups_df = pd.read_csv("../data/user_groups.csv")
    U1 = user_groups_df[user_groups_df["user group"] == 0]["user_id"].tolist()
    U2 = user_groups_df[user_groups_df["user group"] == 1]["user_id"].tolist()
    U3 = user_groups_df[user_groups_df["user group"] == 2]["user_id"].tolist()
    U4 = user_groups_df[user_groups_df["user group"] == 3]["user_id"].tolist()
    beyms = pd.read_csv("../data/beyms.csv")["user_id"].tolist()
    ms = pd.read_csv("../data/ms.csv")["user_id"].tolist()

    ratings_df = pd.read_csv("../data/ratings.csv")

    # compute recommendations
    Random_errors_per_fold, Random_predictions = [], []
    Normal_errors_per_fold, Normal_predictions = [], []
    UserItemAvg_errors_per_fold, UserItemAvg_predictions = [], []
    UserKNN_errors_per_fold, UserKNN_predictions = [], []
    UserKNNAvg_errors_per_fold, UserKNNAvg_predictions = [], []
    NMF_errors_per_fold, NMF_predictions = [], []

    reader = Reader(rating_scale=(1, 1000))
    dataset = Dataset.load_from_df(ratings_df, reader)
    folds_it = KFold(n_splits=5).split(dataset)
    for f, data in enumerate(folds_it):
        trainset, testset = data

        """print("==========================================================================")
        print("[Fold %d], Random" % (f+1))
        print("==========================================================================")
        Random_preds = []
        for rating, r_ in zip(trainset.all_ratings(), np.random.uniform(1, 1000, size=trainset.n_ratings)):
            inner_u, inner_i, r = rating
            u = trainset.to_raw_uid(inner_u)
            i = trainset.to_raw_iid(inner_i)
            details = None
            Random_preds.append((u, i, r, r_, details))

        print(np.mean([np.abs(r-r_) for _, _, r, r_, _ in Random_preds]))
        errors, ttest_results, pairwise_results, mannwhitneyu_results = evaluate_for_each_group(Random_preds, U1, U2, U3, U4, beyms, ms)
        Random_errors_per_fold.append(errors)
        Random_predictions.extend(Random_preds)
        print("Mean Absolute Error: " + str(errors))
        print("t-Test: " + str(ttest_results))
        print("ANOVA: " + str(pairwise_results["ANOVA"]))
        print(pairwise_results["TukeyHSD"])
        print("Mann-Whiteney-U Test: " + str(mannwhitneyu_results))

        print("==========================================================================")
        print("[Fold %d], Normal" % (f + 1))
        print("==========================================================================")
        Normal_preds = NormalPredictor().fit(trainset).test(testset)
        errors, ttest_results, pairwise_results, mannwhitneyu_results = evaluate_for_each_group(Normal_preds, U1, U2,
                                                                                                U3, U4, beyms, ms)
        Normal_errors_per_fold.append(errors)
        Normal_predictions.extend(Normal_preds)
        print("Mean Absolute Error: " + str(errors))
        print("t-Test: " + str(ttest_results))
        print("ANOVA: " + str(pairwise_results["ANOVA"]))
        print(pairwise_results["TukeyHSD"])
        print("Mann-Whiteney-U Test: " + str(mannwhitneyu_results))"""

        print("==========================================================================")
        print("[Fold %d], UserItemAvg" % (f+1))
        print("==========================================================================")
        UserItemAvg_preds = BaselineOnly().fit(trainset).test(testset)
        errors, ttest_results, pairwise_results, mannwhitneyu_results = evaluate_for_each_group(UserItemAvg_preds, U1, U2, U3, U4, beyms, ms)
        UserItemAvg_errors_per_fold.append(errors)
        UserItemAvg_predictions.extend(UserItemAvg_preds)
        print("Mean Absolute Error: " + str(errors))
        print("t-Test: " + str(ttest_results))
        print("ANOVA: " + str(pairwise_results["ANOVA"]))
        print(pairwise_results["TukeyHSD"])
        print("Mann-Whiteney-U Test: " + str(mannwhitneyu_results))

        print("==========================================================================")
        print("[Fold %d], UserKNN" % (f+1))
        print("==========================================================================")
        UserKNN_preds = KNNBasic(k=40, sim_options={"name": "cosine"}).fit(trainset).test(testset)
        errors, ttest_results, pairwise_results, mannwhitneyu_results = evaluate_for_each_group(UserKNN_preds, U1, U2, U3, U4, beyms, ms)
        UserKNN_errors_per_fold.append(errors)
        UserKNN_predictions.extend(UserKNN_preds)
        print("Mean Absolute Error: " + str(errors))
        print("t-Test: " + str(ttest_results))
        print("ANOVA: " + str(pairwise_results["ANOVA"]))
        print(pairwise_results["TukeyHSD"])
        print("Mann-Whiteney-U Test: " + str(mannwhitneyu_results))

        print("==========================================================================")
        print("[Fold %d], UserKNNAvg" % (f+1))
        print("==========================================================================")
        UserKNNAvg_preds = KNNWithMeans(k=40, sim_options={"name": "cosine"}).fit(trainset).test(testset)
        errors, ttest_results, pairwise_results, mannwhitneyu_results = evaluate_for_each_group(UserKNNAvg_preds, U1, U2, U3, U4, beyms, ms)
        UserKNNAvg_errors_per_fold.append(errors)
        UserKNNAvg_predictions.extend(UserKNNAvg_preds)
        print("Mean Absolute Error: " + str(errors))
        print("t-Test: " + str(ttest_results))
        print("ANOVA: " + str(pairwise_results["ANOVA"]))
        print(pairwise_results["TukeyHSD"])
        print("Mann-Whiteney-U Test: " + str(mannwhitneyu_results))

        print("==========================================================================")
        print("[Fold %d], NMF" % (f+1))
        print("==========================================================================")
        NMF_preds = NMF(n_factors=15).fit(trainset).test(testset)
        errors, ttest_results, pairwise_results, mannwhitneyu_results = evaluate_for_each_group(NMF_preds, U1, U2, U3, U4, beyms, ms)
        NMF_errors_per_fold.append(errors)
        NMF_predictions.extend(NMF_preds)
        print("Mean Absolute Error: " + str(errors))
        print("t-Test: " + str(ttest_results))
        print("ANOVA: " + str(pairwise_results["ANOVA"]))
        print(pairwise_results["TukeyHSD"])
        print("Mann-Whiteney-U Test: " + str(mannwhitneyu_results))


    print("[MAE over all folds] UserItemAvg: " + str(mean_of_dictionaries(*UserItemAvg_errors_per_fold)))
    print("[MAE over all folds] UserKNN: " + str(mean_of_dictionaries(*UserKNN_errors_per_fold)))
    print("[MAE over all folds] UserKNNAvg: " + str(mean_of_dictionaries(*UserKNNAvg_errors_per_fold)))
    print("[MAE over all folds] NMF: " + str(mean_of_dictionaries(*NMF_errors_per_fold)))

    print("============================================================")

    errors, ttest_results, pairwise_results, mannwhitneyu_results = \
        evaluate_for_each_group(Random_predictions, U1, U2, U3, U4, beyms, ms)
    print("Mean Absolute Error: " + str(errors))
    print("t-Test: " + str(ttest_results))
    print("ANOVA: " + str(pairwise_results["ANOVA"]))
    print(pairwise_results["TukeyHSD"])
    print("Mann-Whiteney-U Test: " + str(mannwhitneyu_results))

    errors, ttest_results, pairwise_results, mannwhitneyu_results = \
        evaluate_for_each_group(Normal_predictions, U1, U2, U3, U4, beyms, ms)
    print("Mean Absolute Error: " + str(errors))
    print("t-Test: " + str(ttest_results))
    print("ANOVA: " + str(pairwise_results["ANOVA"]))
    print(pairwise_results["TukeyHSD"])
    print("Mann-Whiteney-U Test: " + str(mannwhitneyu_results))

    errors, ttest_results, pairwise_results, mannwhitneyu_results = \
        evaluate_for_each_group(UserItemAvg_predictions, U1, U2, U3, U4, beyms, ms)
    print("Mean Absolute Error: " + str(errors))
    print("t-Test: " + str(ttest_results))
    print("ANOVA: " + str(pairwise_results["ANOVA"]))
    print(pairwise_results["TukeyHSD"])
    print("Mann-Whiteney-U Test: " + str(mannwhitneyu_results))

    errors, ttest_results, pairwise_results, mannwhitneyu_results = \
        evaluate_for_each_group(UserKNN_predictions, U1, U2, U3, U4, beyms, ms)
    print("Mean Absolute Error: " + str(errors))
    print("t-Test: " + str(ttest_results))
    print("ANOVA: " + str(pairwise_results["ANOVA"]))
    print(pairwise_results["TukeyHSD"])
    print("Mann-Whiteney-U Test: " + str(mannwhitneyu_results))

    errors, ttest_results, pairwise_results, mannwhitneyu_results = \
        evaluate_for_each_group(UserKNNAvg_predictions, U1, U2, U3, U4, beyms, ms)
    print("Mean Absolute Error: " + str(errors))
    print("t-Test: " + str(ttest_results))
    print("ANOVA: " + str(pairwise_results["ANOVA"]))
    print(pairwise_results["TukeyHSD"])
    print("Mann-Whiteney-U Test: " + str(mannwhitneyu_results))

    errors, ttest_results, pairwise_results, mannwhitneyu_results = \
        evaluate_for_each_group(NMF_predictions, U1, U2, U3, U4, beyms, ms)
    print("Mean Absolute Error: " + str(errors))
    print("t-Test: " + str(ttest_results))
    print("ANOVA: " + str(pairwise_results["ANOVA"]))
    print(pairwise_results["TukeyHSD"])
    print("Mann-Whiteney-U Test: " + str(mannwhitneyu_results))







