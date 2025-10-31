import numpy as np
from Iq_analyze import calc_PQ

_RNORM = np.random.normal(0.0, 1.0, 10_000)  # standard normal, fixed
_RUNIF = np.random.uniform(-1.0, 1.0, 10_000)  # for uniform case


def gen_Rs_from_distribution(R0, sigma, pdType):
    if pdType == "uniform":  # uniform
        return R0 * (1 + sigma * _RUNIF)
    elif pdType == "normal":  # normal
        return R0 * (1 + sigma * _RNORM)
    elif pdType == "lognormal":  # log-normal
        return R0 * np.exp(sigma * _RNORM)
    else:
        raise ValueError("Unknown polydisperse type")


def calc_rod_Rg2(L, D):
    # $k^2 = \frac{ \frac{L^3}{12} + L r^2 + \frac{4}{5} r^3 + \frac{L^2 r}{3} }{ L + \frac{4}{3} r }$
    r = 0.5 * D
    Rg2 = ((L**3 / 12) + L * r**2 + (4 / 5) * r**3 + (L**2 * r / 3)) / (L + (4 / 3) * r)
    return Rg2


def calc_HS_PY_SQ(Q, Reff, eta):
    # S_PY(Q) = 1/(1+24 phi G(x,phi)/x)

    # https://en.wikipedia.org/wiki/Percus–Yevick_approximation
    """
    /* abridged from sasmodels/models/hardsphere.c */
    double py(double qr, double eta) {
        const double a = pow(1+2*eta, 2)/pow(1-eta, 4);
        const double b = -6*eta*pow(1+eta/2, 2)/pow(1-eta, 4);
        const double c = 0.5*eta*a;
        const double x = 2*qr;                 //  x = 2 q R_eff
        const double x2 = x*x;

        const double G =
            a/x2   *(sin(x)-x*cos(x))
            + b/x2/x *(2*x*sin(x)+(2-x2)*cos(x)-2)
            + c/pow(x,5)*(-pow(x,4)*cos(x)
                + 4*((3*x2-6)*cos(x) + x*(x2-6)*sin(x) + 6));

        return 1.0/(1.0 + 24.0*eta*G/x);
    }
    """
    a = (1 + 2 * eta) ** 2 / (1 - eta) ** 4
    b = -6 * eta * (1 + eta / 2) ** 2 / (1 - eta) ** 4
    c = 0.5 * eta * a
    x = 2 * Q * Reff  # x = 2 q R_eff
    x2 = x * x

    G = (
        a / x2 * (np.sin(x) - x * np.cos(x))
        + b / x2 / x * (2 * x * np.sin(x) + (2 - x2) * np.cos(x) - 2)
        + c / x**5 * (-(x**4) * np.cos(x) + 4 * ((3 * x2 - 6) * np.cos(x) + x * (x2 - 6) * np.sin(x) + 6))
    )

    SQ_PY = 1.0 / (1.0 + 24.0 * eta * G / x)
    return SQ_PY


def calc_PY_IQ(Qs, eta, meanL, sigmaD, pdType):
    R0 = 0.5
    Rs = gen_Rs_from_distribution(R0, sigmaD, pdType)
    Rg2s = calc_rod_Rg2(L=meanL, D=2 * Rs)
    Rsp2s = Rg2s * 5 / 3  # convert Rg to sphere equivalent R
    # Reff is mean_volume weighted Rsp, and rescale to sphere Reff
    Reff = np.mean(Rsp2s ** (3 / 2)) ** (1 / 3)

    SQ_PY = calc_HS_PY_SQ(Qs, Reff, eta)

    PQ = calc_PQ(Qs, meanL, sigmaD, pdType)
    IQ = SQ * PQ
    return IQ
