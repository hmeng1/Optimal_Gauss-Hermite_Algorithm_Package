"""Damped Newton / LM-style method using EXACT Hessian of F(x)=0.5||r(x)||^2."""

import numpy as np
from scipy.linalg import cholesky, cho_solve

from ._common import merge_defaults, gain_ratio, damping_diag as _damping_diag


def nls_lm_exact_h(resfun, x0, opts=None):
    """Minimize F(x) = 0.5||r||^2 using exact Hessian with LM damping.

    resfun: callable (r, J, H) = resfun(x) where H is exact Hessian of F.
    """
    if opts is None:
        opts = {}
    opts = merge_defaults(opts, _DEFAULTS)

    x = np.asarray(x0, dtype=float).ravel()
    n = len(x)
    diag_idx = np.arange(0, n * n, n + 1)
    funcCount = 0

    r, J, H = resfun(x)
    funcCount += 1
    r = np.asarray(r).ravel()
    F = 0.5 * (r @ r)
    g = J.T @ r
    gInf = np.max(np.abs(g))
    F0 = F

    H = 0.5 * (H + H.T)
    diagH = _safe_diag_hess(H)

    mu = opts['mu0'] * max(1, np.max(diagH))
    mu = min(max(mu, opts['muMin']), opts['muMax'])
    nu = opts['nu0']

    info = {
        'exitflag': 1, 'stopCode': 4,
        'message': f"Stopped: maxIter reached (maxIter={opts['maxIter']}).",
        'iter': 0, 'funcCount': funcCount,
        'stopIter': 0, 'stopTrial': 0, 'stopMode': 'exactH',
        'stopResnorm': r @ r, 'stopOptInf': gInf, 'stopMu': mu,
        'fHist': [F], 'resnormHist': [r @ r], 'gInfHist': [gInf], 'muHist': [mu],
        'rhoHist': [], 'stepNormHist': [], 'acceptedHist': [],
    }

    _display_header(opts['Display'])

    for k in range(1, opts['maxIter'] + 1):
        info['iter'] = k

        if gInf <= opts['tolGrad']:
            info = _set_stop(info, 0, 1,
                             f"Stopped: ||g||_inf <= tolGrad ({opts['tolGrad']:.3e}).",
                             k, 0, r @ r, gInf, mu)
            break

        x_base = x
        F_base = F
        g_base = g
        H_base = H
        Ddiag = _damping_diag(_safe_diag_hess(H_base), opts['damping'])

        accepted = False

        for t in range(1, opts['maxTrials'] + 1):
            A = H_base.copy()
            A.flat[diag_idx] += mu * Ddiag
            A = 0.5 * (A + A.T)

            try:
                L = cholesky(A, lower=True)
                p = -cho_solve((L, True), g_base)
                pflag = 0
            except np.linalg.LinAlgError:
                pflag = 1

            if pflag != 0:
                mu = min(max(mu * nu, opts['muMin']), opts['muMax'])
                nu = min(opts['nuMax'], 2 * nu)
                continue

            stepNorm = np.linalg.norm(p)
            relStep = stepNorm / max(1, np.linalg.norm(x_base))

            if relStep <= opts['tolStep']:
                accepted = True
                x = x_base
                info = _set_stop(info, 0, 2,
                                 f"Stopped: relative step <= tolStep ({opts['tolStep']:.3e}).",
                                 k, t, 2 * F_base, gInf, mu)
                break

            x_try = x_base + p
            r_try, J_try, H_try = resfun(x_try)
            funcCount += 1
            r_try = np.asarray(r_try).ravel()
            F_try = 0.5 * (r_try @ r_try)

            model = F_base + (g_base @ p) + 0.5 * (p @ (A @ p))
            pred = F_base - model
            act = F_base - F_try
            rho = gain_ratio(act, pred, F_try)

            if rho > opts['minAcceptRho'] and np.isfinite(rho):
                accepted = True
                x = x_try
                r = r_try
                J = J_try
                H = 0.5 * (H_try + H_try.T)
                F = F_try
                g = J.T @ r
                gInf = np.max(np.abs(g))

                mu = mu * max(1 / 3, 1 - (2 * rho - 1)**3)
                mu = min(max(mu, opts['muMin']), opts['muMax'])
                nu = opts['nu0']

                info['rhoHist'].append(rho)
                info['stepNormHist'].append(stepNorm)
                info['acceptedHist'].append(True)
                break

            else:
                mu = min(max(mu * nu, opts['muMin']), opts['muMax'])
                nu = min(opts['nuMax'], 2 * nu)

        info['funcCount'] = funcCount

        if info['exitflag'] == 0 and info['stopCode'] == 2:
            break

        if not accepted:
            info = _set_stop(info, 3, 6,
                             f"Stopped: no acceptable step after {opts['maxTrials']} trials.",
                             k, opts['maxTrials'], 2 * F_base, gInf, mu)
            break

        info['fHist'].append(F)
        info['resnormHist'].append(r @ r)
        info['gInfHist'].append(gInf)
        info['muHist'].append(mu)

        relChange = abs(F_base - F) / max(1, F0)
        if relChange <= opts['tolCost']:
            info = _set_stop(info, 0, 3,
                             f"Stopped: relative cost change <= tolCost ({opts['tolCost']:.3e}).",
                             k, 0, r @ r, gInf, mu)
            break

    if info['exitflag'] == 1 and info['iter'] >= opts['maxIter']:
        info = _set_stop(info, 1, 4,
                         f"Stopped: maxIter reached (maxIter={opts['maxIter']}).",
                         info['iter'], 0, r @ r, gInf, mu)

    _display_final(opts['Display'], info)
    return x, info


_DEFAULTS = {
    'maxIter': 100, 'maxTrials': 10,
    'tolGrad': 1e-8, 'tolStep': 1e-10, 'tolCost': 1e-12,
    'mu0': 1e-2, 'muMin': 1e-15, 'muMax': 1e15,
    'nu0': 2, 'nuMax': 1e15,
    'minAcceptRho': 1e-3,
    'damping': 'diagHabs',
    'Display': 'off',
}


def _safe_diag_hess(H):
    d = np.abs(np.diag(H))
    d[~np.isfinite(d) | (d <= 0)] = 1
    return d



def _set_stop(info, exitflag, stopCode, msg, it, trial, resnorm, optInf, mu):
    info['exitflag'] = exitflag
    info['stopCode'] = stopCode
    info['message'] = msg
    info['stopIter'] = it
    info['stopTrial'] = trial
    info['stopMode'] = 'exactH'
    info['stopResnorm'] = resnorm
    info['stopOptInf'] = optInf
    info['stopMu'] = mu
    return info


def _display_header(Display):
    if Display == 'off':
        return
    if Display == 'iter':
        print(f"\n{'Iter':>10} {'Trial':>6} {'Func':>10} {'Resnorm':>14} {'optInf':>14} {'Lambda':>12} {'Step':>12}  {'Status':<10}")
        print('-' * 92)


def _display_final(Display, info):
    if Display in ('off', 'iter'):
        return
    print(f"\nStop reason: {info['message']}")
    print(f"exitflag={info['exitflag']}, stopCode={info['stopCode']}")
