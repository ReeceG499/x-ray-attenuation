import pandas as pd
from scipy.interpolate import interp1d

def load_mu_data(csv_path, density, energy_range_KeV):
    df = pd.read_csv(csv_path, names=["Energy_MeV", "Mu_Rho"], header=None)
    df["Energy_keV"] = df["Energy_MeV"] * 1000
    df = df.groupby("Energy_keV", as_index=False).mean()
    df["Mu"] = df["Mu_Rho"] * density

    interp_mu = interp1d(df["Energy_keV"], df["Mu"], bounds_error=False, fill_value="extrapolate")
    return interp_mu(energy_range_KeV)