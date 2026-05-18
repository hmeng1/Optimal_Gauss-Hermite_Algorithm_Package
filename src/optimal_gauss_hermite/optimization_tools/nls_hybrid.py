"""Hybrid GN(far) + exact-H(near), NO normal equations."""

import numpy as np
from scipy.linalg import cholesky, cho_solve

from ._common import merge_defaults, gain_ratio, diag_jtj as _diag_jtj, damping_diag as _damping_diag


def nls_hybrid(resfun, x0, opts=None):
    """Minimize F(x) = 0.5||r(x)||^2 with hybrid far/near mode.

    resfun supports: r=resfun(x), (r,J)=resfun(x), (r,J,H)=resfun(x).
    In Python, resfun always returns (r, J, H).

    FAR mode: augmented LS step (QR/backslash), no normal equations.
    NEAR mode: exact-H damped Newton (Cholesky), no normal equations.
    """
    if opts is None:
        opts = {}
    opts = merge_defaults(opts, _DEFAULTS)

    x = np.asarray(x0, dtype=float).ravel()
    n = len(x)
    diag_idx = np.arange(0, n * n, n + 1)

    mu = opts['mu0']
    nu = opts['nu0']

    is_far_mode = True
    good_count = 0
    near_rejects = 0
    funcCount = 0

    info = {
        'exitflag': 1, 'stopCode': 4,
        'message': f"Stopped: maxIter reached (maxIter={opts['maxIter']}).",
        'iter': 0, 'funcCount': 0,
        'stopIter': 0, 'stopTrial': 0, 'stopMode': 'init',
        'stopResnorm': np.nan, 'stopOptInf': np.nan, 'stopMu': np.nan,
        'modeHist': [], 'fHist': [], 'gInfHist': [], 'muHist': [],
        'rhoHist': [], 'stepNormHist': [], 'acceptedHist': [],
    }

    # Initial eval
    r, J, _ = resfun(x)
    funcCount += 1
    r = np.asarray(r).ravel()
    F = 0.5 * (r @ r)
    g = J.T @ r
    gInf = np.max(np.abs(g))

    info['fHist'].append(F)
    info['gInfHist'].append(gInf)
    info['muHist'].append(mu)

    _display_header(opts['Display'])

    for k in range(1, opts['maxIter'] + 1):
        info['iter'] = k
        mode_name = 'far' if is_far_mode else 'near'
        info['modeHist'].append(mode_name)

        if gInf <= opts['tolGrad']:
            info = _set_stop(info, 0, 1,
                             f"Stopped: ||g||_inf <= tolGrad ({opts['tolGrad']:.3e}).",
                             k, 0, mode_name, 2 * F, gInf, mu)
            break

        accepted = False

        is_near_now = (gInf <= opts['switchGrad']) and (good_count >= opts['switchGoodCount'])
        if is_near_now:
            is_far_mode = False

        x_base = x
        F_base = F
        r_base = r
        J_base = J
        g_base = g
        gInf_base = gInf

        if is_far_mode:
            have_near_base = False
            rH = None
            JH = None
            H = None
            DdiagH = None
            F_near_base = np.nan
            g_near_base = None
            gInf_near_base = np.nan

        if is_far_mode:
            diagJTJ = _diag_jtj(J_base)
            Ddiag = _damping_diag(diagJTJ, opts['dampingGN'])
            JtJ = J_base.T @ J_base

        for t in range(1, opts['maxTrials'] + 1):
            mode_name = 'far' if is_far_mode else 'near'

            if is_far_mode:
                p, ok = _step_far_aug(J_base, r_base, mu, Ddiag, JtJ)
                if not ok or not np.all(np.isfinite(p)):
                    mu = min(max(mu * nu, opts['muMin']), opts['muMax'])
                    nu = min(opts['nuMax'], 2 * nu)
                    continue

                stepNorm = np.linalg.norm(p)
                xBaseNorm = max(1, np.linalg.norm(x_base))

                if stepNorm / xBaseNorm <= opts['tolStep']:
                    accepted = True
                    info = _set_stop(info, 0, 2,
                                     f"Stopped: relative step <= tolStep ({opts['tolStep']:.3e}).",
                                     k, t, mode_name, 2 * F_base, gInf_base, mu)
                    break

                x_try = x_base + p
                r_try, J_try, _ = resfun(x_try)
                funcCount += 1
                r_try = np.asarray(r_try).ravel()
                F_try = 0.5 * (r_try @ r_try)

                quad = p @ (JtJ @ p)
                model = F_base + (g_base @ p) + 0.5 * quad + 0.5 * mu * np.sum(Ddiag * p**2)
                pred = F_base - model
                act = F_base - F_try
                rho = gain_ratio(act, pred, F_try)

                if rho > opts['minAcceptRho']:
                    accepted = True
                    x = x_try
                    r = r_try
                    J = J_try
                    F = F_try
                    g = J.T @ r
                    gInf = np.max(np.abs(g))

                    if rho >= opts['switchGoodRho']:
                        good_count += 1
                    else:
                        good_count = 0
                    near_rejects = 0

                    mu = mu * max(1 / 3, 1 - (2 * rho - 1)**3)
                    mu = min(max(mu, opts['muMin']), opts['muMax'])
                    nu = opts['nu0']

                    info['rhoHist'].append(rho)
                    info['stepNormHist'].append(stepNorm)
                    info['acceptedHist'].append(True)
                    break

                else:
                    good_count = 0
                    mu = min(max(mu * nu, opts['muMin']), opts['muMax'])
                    nu = min(opts['nuMax'], 2 * nu)

            else:
                # NEAR mode
                use_lm_near = (opts['nearMethod'] == 'levenberg-marquardt')

                if not have_near_base:
                    if use_lm_near:
                        # LM in NEAR: reuse base data already at hand, no extra resfun call
                        rH = r_base
                        JH = J_base
                        F_near_base = F_base
                        g_near_base = g_base
                        gInf_near_base = gInf_base
                    else:
                        rH, JH, H = resfun(x_base)
                        funcCount += 1
                        rH = np.asarray(rH).ravel()
                        H = 0.5 * (H + H.T)
                        F_near_base = 0.5 * (rH @ rH)
                        g_near_base = JH.T @ rH
                        gInf_near_base = np.max(np.abs(g_near_base))

                    have_near_base = True

                if use_lm_near:
                    diagJTJn = _diag_jtj(JH)
                    DdiagH = _damping_diag(diagJTJn, opts['dampingGN'])
                    JtJn = JH.T @ JH

                    p, ok = _step_far_aug(JH, rH, mu, DdiagH, JtJn)
                    A_damped = None  # model will use JtJn directly
                    if not ok or not np.all(np.isfinite(p)):
                        near_rejects += 1
                        mu = min(max(mu * nu, opts['muMin']), opts['muMax'])
                        nu = min(opts['nuMax'], 2 * nu)
                        if near_rejects >= opts['backCholFails']:
                            is_far_mode = True
                            good_count = 0
                            break
                        continue
                else:
                    modeH = opts['dampingH']
                    if modeH == 'identity':
                        diagBase = np.ones(n)
                    elif modeH == 'diagJTJ':
                        diagBase = _diag_jtj(JH)
                    elif modeH == 'diagHabs':
                        diagBase = np.abs(np.diag(H))
                        diagBase[~np.isfinite(diagBase) | (diagBase <= 0)] = 1
                    elif modeH == 'mix':
                        diagBase = np.abs(np.diag(H))
                        diagBase[~np.isfinite(diagBase) | (diagBase <= 0)] = 1
                        diagBase = diagBase + _diag_jtj(JH)
                    else:
                        raise ValueError(f"Unknown dampingH: {modeH}")
                    DdiagH = _damping_diag(diagBase, modeH)

                    p, cholOK, A_damped = _step_near_exact_h(H, g_near_base, mu, DdiagH, diag_idx)
                    if not cholOK or not np.all(np.isfinite(p)):
                        near_rejects += 1
                        mu = min(max(mu * nu, opts['muMin']), opts['muMax'])
                        nu = min(opts['nuMax'], 2 * nu)
                        if near_rejects >= opts['backCholFails']:
                            is_far_mode = True
                            good_count = 0
                            break
                        continue

                stepNorm = np.linalg.norm(p)
                xBaseNorm = max(1, np.linalg.norm(x_base))

                if stepNorm / xBaseNorm <= opts['tolStep']:
                    accepted = True
                    x = x_base
                    r = rH
                    J = JH
                    F = F_near_base
                    g = g_near_base
                    gInf = gInf_near_base

                    info = _set_stop(info, 0, 2,
                                     f"Stopped: relative step <= tolStep ({opts['tolStep']:.3e}).",
                                     k, t, mode_name, 2 * F, gInf, mu)
                    break

                x_try = x_base + p
                H_try = None
                if use_lm_near:
                    r_try, J_try, _ = resfun(x_try)
                    funcCount += 1
                    r_try = np.asarray(r_try).ravel()
                    F_try = 0.5 * (r_try @ r_try)

                    quad = p @ (JtJn @ p)
                    model = F_near_base + (g_near_base @ p) + 0.5 * quad + 0.5 * mu * np.sum(DdiagH * p**2)
                else:
                    r_try, J_try, H_try = resfun(x_try)
                    funcCount += 1
                    r_try = np.asarray(r_try).ravel()
                    F_try = 0.5 * (r_try @ r_try)

                    model = F_near_base + (g_near_base @ p) + 0.5 * (p @ (A_damped @ p))

                pred = F_near_base - model
                act = F_near_base - F_try
                rho = gain_ratio(act, pred, F_try)

                if rho > opts['minAcceptRho']:
                    accepted = True
                    x = x_try
                    r = r_try
                    J = J_try
                    F = F_try
                    g = J.T @ r
                    gInf = np.max(np.abs(g))
                    # Prepopulate NEAR base for next iteration
                    rH = r_try
                    JH = J_try
                    F_near_base = F_try
                    g_near_base = g
                    gInf_near_base = gInf
                    have_near_base = True
                    if not use_lm_near:
                        H = 0.5 * (H_try + H_try.T)

                    if rho >= opts['switchGoodRho']:
                        good_count += 1
                    else:
                        good_count = 0
                    near_rejects = 0

                    mu = mu * max(1 / 3, 1 - (2 * rho - 1)**3)
                    mu = min(max(mu, opts['muMin']), opts['muMax'])
                    nu = opts['nu0']

                    info['rhoHist'].append(rho)
                    info['stepNormHist'].append(stepNorm)
                    info['acceptedHist'].append(True)
                    break

                else:
                    near_rejects += 1
                    good_count = 0
                    mu = min(max(mu * nu, opts['muMin']), opts['muMax'])
                    nu = min(opts['nuMax'], 2 * nu)

                    if near_rejects >= opts['backRejects']:
                        is_far_mode = True
                        break
                    continue

        info['funcCount'] = funcCount

        if not accepted:
            if is_far_mode:
                info['fHist'].append(F)
                info['gInfHist'].append(gInf)
                info['muHist'].append(mu)
                continue
            mode_name = 'far' if is_far_mode else 'near'
            info = _set_stop(info, 3, 6,
                             f"Stopped: no acceptable step after {opts['maxTrials']} trials.",
                             k, opts['maxTrials'], mode_name, 2 * F_base, gInf_base, mu)
            break

        info['fHist'].append(F)
        info['gInfHist'].append(gInf)
        info['muHist'].append(mu)

        if info['exitflag'] == 1:
            relChange = abs(F_base - F) / max(1, info['fHist'][0])
            if relChange <= opts['tolCost']:
                info = _set_stop(info, 0, 3,
                                 f"Stopped: relative cost change <= tolCost ({opts['tolCost']:.3e}).",
                                 k, 0, mode_name, 2 * F, gInf, mu)
                break
        else:
            break

    if info['exitflag'] == 1 and info['iter'] >= opts['maxIter']:
        mode_name = 'far' if is_far_mode else 'near'
        info = _set_stop(info, 1, 4,
                         f"Stopped: maxIter reached (maxIter={opts['maxIter']}).",
                         info['iter'], 0, mode_name, 2 * F, gInf, mu)

    _display_final(opts['Display'], info)
    return x, info


# --------------- step functions ---------------
def _step_far_aug(J, r, mu, Ddiag, JtJ=None):
    n = J.shape[1]
    g = J.T @ r
    if JtJ is None:
        JtJ = J.T @ J
    A = JtJ.copy()
    if mu > 0:
        A.flat[::n + 1] += mu * Ddiag
    try:
        L = cholesky(A, lower=True)
        p = cho_solve((L, True), -g)
        return p, True
    except np.linalg.LinAlgError:
        return np.zeros(n), False


def _step_near_exact_h(H, g, mu, Ddiag, diag_idx):
    n = len(g)
    A = H.copy()
    A.flat[diag_idx] += mu * Ddiag
    A = 0.5 * (A + A.T)
    try:
        L = cholesky(A, lower=True)
        p = -cho_solve((L, True), g)
        return p, True, A
    except np.linalg.LinAlgError:
        return np.zeros(n), False, None


# --------------- utilities ---------------


def _set_stop(info, exitflag, stopCode, msg, it, trial, mode, resnorm, optInf, mu):
    info['exitflag'] = exitflag
    info['stopCode'] = stopCode
    info['message'] = msg
    info['stopIter'] = it
    info['stopTrial'] = trial
    info['stopMode'] = str(mode)
    info['stopResnorm'] = resnorm
    info['stopOptInf'] = optInf
    info['stopMu'] = mu
    return info


_DEFAULTS = {
    'maxIter': 100, 'maxTrials': 10,
    'tolGrad': 1e-8, 'tolStep': 1e-10, 'tolCost': 1e-12,
    'mu0': 1e-2, 'muMin': 1e-15, 'muMax': 1e15,
    'nu0': 2, 'nuMax': 1e15,
    'minAcceptRho': 1e-3,
    'switchGrad': 1e-6,
    'switchGoodRho': 0.75,
    'switchGoodCount': 2,
    'backRejects': 2,
    'backCholFails': 2,
    'dampingGN': 'diagJTJ',
    'dampingH': 'diagHabs',
    'nearMethod': 'levenberg-marquardt',
    'Display': 'off',
}


def _display_header(Display):
    if Display == 'off':
        return
    headers = ['Iter', 'Trial', 'Func', 'Resnorm', 'optInf', 'Lambda', 'rho', 'Step', 'Status', 'Mode']
    print('\n' + ''.join(f'{h:>10}' for h in headers))
    print('-' * 100)


def _display_final(Display, info):
    if Display == 'off':
        return
    print(f"\nStop reason: {info['message']}")
    print(f"exitflag={info['exitflag']}, stopCode={info['stopCode']}")
