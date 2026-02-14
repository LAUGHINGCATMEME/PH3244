import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



import labtools
labtools.import_data()          # reads CSVs from ./data/
labtools.unit_check()
labtools.unit_converter()
labtools.make_parsed_data()     # creates parsed_data/<OBS>_SI_PARSED.csv



# add Voltage_sq and Voltage_sq_err for observation named "for I = 1 Amp"
labtools.propagate_square("for I = 1 Amp", "Voltage")
# result is in labtools. for I = 1 Amp  (sanitized attribute also available)
# columns Voltage_sq and Voltage_sq_err appear
labtools.propagate_log("for I = 1 Amp", "Voltage", out_col="lnV")


df = labtools.OBSERVATIONS["for I = 1 Amp"]
print(df)

def d_func(rhs, lhs):
    return rhs - lhs

# optional: provide jacobian to be more accurate & faster:
def d_jac(rhs, lhs):
    # dD/drhs = 1, dD/dlhs = -1
    return np.vstack((1.0 ,-1.0)).T   # shape (n, 2)
labtools.propagate_function("for I = 1 Amp",
                            ["Voltage", "rhs"],
                            r_func,
                            "D",
                            jacobian=r_jac)


# simple linear (default)
res = labtools.fit_odr("for I = 1 Amp", x_col="rhs", y_col="lhs",
                       x_err_col="rhs_err", y_err_col="lhs_err")
# res contains params, param_err, etc.
imgpath = labtools.plot_with_fit("for I = 1 Amp", "rhs", "lhs",
                                 x_err_col="rhs_err", y_err_col="lhs_err",
                                 fit_result=res,
                                 title="rhs vs lhs for for I = 1 Amp")

