import pandas as pd
import numpy as np

rows = 100000

data = {
    "Day_of_Week": np.random.randint(1,8,rows),
    "Season": np.random.randint(1,5,rows),
    "Total_Beds": np.random.randint(100,500,rows),
    "Emergency_Admissions": np.random.randint(10,80,rows),
    "Scheduled_Admissions": np.random.randint(5,40,rows),
    "Discharges": np.random.randint(5,60,rows),
    "Avg_Length_of_Stay": np.random.randint(2,10,rows),
    "ICU_Admissions": np.random.randint(1,20,rows),
    "Holiday": np.random.randint(0,2,rows),
    "Staff_Availability": np.random.randint(1,20,rows),
    "Local_Disease_Index": np.random.randint(0,5,rows),
    "Festival_Holiday_Index": np.random.randint(0,1,rows),
    "Population_Density": np.random.randint(100000,105000,rows),
    "Nearby_Hospital_Capacity": np.random.randint(3,5,rows)
}

df = pd.DataFrame(data)

df["Bed_Occupancy_Rate"] = (
    df["Emergency_Admissions"] +
    df["Scheduled_Admissions"] -
    df["Discharges"]
) / df["Total_Beds"] * 100

df.to_csv("bed_occupancy_dataset.csv", index=False)