import os, sys; sys.path.insert(0, os.path.abspath("."))
from scenarios.NSP_QR_cell import run

params_available_NV = {"P_LINK": 5 * 10**-2,
                       "f_clock": 50 * 10**6,
                       "T_DP": 10 * 10**-3}
params_available_SiV = {"P_LINK": 5 * 10**-2,
                        "f_clock": 30 * 10**6,
                        "T_DP": 1 * 10**-3}
params_available_qdots = {"P_LINK": 10 * 10**-2,
                          "f_clock": 1000 * 10**6,
                          "T_DP": 0.003 * 10**-3}
params_available_Ca = {"P_LINK": 0.4 * 10**-2,
                       "f_clock": 0.06 * 10**6,
                       "T_DP": 0.8 * 10**-3}
params_available_Rb = {"P_LINK": 70 * 10**-2,
                       "f_clock": 5 * 10**6,
                       "T_DP": 100 * 10**-3}
params_future_NV = {"P_LINK": 50 * 10**-2,
                    "f_clock": 250 * 10**6,
                    "T_DP": 10000 * 10**-3}
params_future_SiV = {"P_LINK": 50 * 10**-2,
                     "f_clock": 500 * 10**6,
                     "T_DP": 100 * 10**-3}
params_future_qdots = {"P_LINK": 60 * 10**-2,
                       "f_clock": 1000 * 10**6,
                       "T_DP": 0.3 * 10**-3}
params_future_Ca = {"P_LINK": 10 * 10**-2,
                    "f_clock": 1 * 10**6,
                    "T_DP": 1 * 10**-3}
params_future_Rb = {"P_LINK": 70 * 10**-2,
                    "f_clock": 100 * 10**6,
                    "T_DP": 1000 * 10**-3}

available_params = [params_available_NV, params_available_SiV, params_available_qdots, params_available_Ca, params_available_Rb]
future_params = [params_future_NV, params_future_SiV, params_future_qdots, params_future_Ca, params_future_Rb]
