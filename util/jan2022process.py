import pandas as pd
from prophet import Prophet


def link_store(row):
    """
    Usage: r1 = df_small.apply(link_store, axis=1)
    :param row:
    :return:
    """
    sep = "___"
    store_name = row["country"] + sep + row["store"] + sep + row["product"].replace(" ", "x")
    row["store_name"] = store_name
    return row


def unlink_store(row):
    sep = "___"
    store_name = row["store_name"].split(sep)
    row["country"] = store_name[0]
    row["store"] = store_name[1]
    row["product"] = store_name[2].replace("x", " ")
    return row


def process_a_store(df_pivot1, df_date, store_name, model_dict, train_dict, forcase_dict):
    df_process = pd.DataFrame(data={"ds": df_pivot1.index, "y": df_pivot1[store_name]})
    model = Prophet()
    model.fit(df_process)
    df_test_date = df_date[["date"]].drop_duplicates()
    df_test_date.columns = ["ds"]
    df_forecast = model.predict(df_test_date)
    model_dict[store_name] = model
    train_dict[store_name] = df_process
    forcase_dict[store_name] = df_forecast
    return df_forecast
