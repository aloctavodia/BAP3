import pandas as pd

datab = pd.read_csv("bikes_full.csv")[["count", "hour", "temperature", "humidity", "windspeed", "weekday"]][::50]
t_min=-8
t_max=+39
datab["temperature"] = datab["temperature"] * (t_max-t_min) + t_min
datab.rename(columns={"count":"rented"}, inplace=True)
datab.to_csv("bikes.csv", index=False, float_format="%.4g")