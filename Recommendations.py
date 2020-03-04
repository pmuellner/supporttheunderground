import pandas as pd
import numpy as np
import sqlalchemy
from sklearn.preprocessing import MinMaxScaler
from surprise import Reader, Dataset, KNNBasic, BaselineOnly, KNNWithMeans, NMF
from surprise.model_selection import KFold

#
# Generation of Rating Dataset
#
beyms = pd.read_csv("data/beyms.csv")["user_id"].tolist()
ms = pd.read_csv("data/ms.csv")["user_id"].tolist()

db_options_df = pd.read_csv("db_credentials.txt", sep="=", header=None)
db_options_df.columns = ["variable", "value"]
db_options_df = db_options_df.apply(lambda col: col.str.strip())
db_options_df.set_index("variable", inplace=True)
db_options = db_options_df["value"].to_dict()
connection = sqlalchemy.create_engine('mysql+pymysql://' + db_options["DB_USERNAME"] + ":" + db_options["DB_PW"] + db_options["DB_PATH"])

events_df = pd.read_sql(con=connection, sql="SELECT user_id, track_id FROM events WHERE user_id IN " + str(tuple(beyms+ms)))
playcounts_df = events_df.groupby(["user_id", "track_id"]).size().reset_index()
playcounts_df.columns = ["user_id", "track_id", "playcount"]

ratings_df = pd.DataFrame()
for user_id, data in playcounts_df.groupby("user_id"):
    ratings = MinMaxScaler(feature_range=(0, 1000)).fit_transform(data["playcount"].values.reshape(-1, 1).astype(float))
    new_rows = data[["user_id", "track_id"]].copy()
    new_rows["rating"] = ratings
    ratings_df = ratings_df.append(new_rows)
ratings_df.columns = ["user_id", "item_id", "rating"]