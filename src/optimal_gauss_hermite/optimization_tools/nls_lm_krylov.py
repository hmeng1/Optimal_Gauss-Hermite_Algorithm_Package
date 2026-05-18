"""Levenberg-Marquardt with multi-trial accept/reject and selectable inner solver."""

import numpy as np
from scipy.linalg import cholesky, cho_solve

from ._common import merge_defaults, diag_jtj as _diag_jtj, damping_diag as _damping_diag


def nls_lm_krylov(resfun, x0, opts=None):
    """Minimize F(x) = 0.5 * ||r(x)||^2 using LM with PCG/LSQR inner solvers.

    resfun: callable (r, J) = resfun(x)
    """
    if opts is None:
        opts = {}
    opts = merge_defaults(opts, _DEFAULTS)

    x = np.asarray(x0, dtype=float).ravel()
    n = len(x)

    if opts['pcgMaxIter'] is None:
        opts['pcgMaxIter'] = min(200, n)
    if opts['lsqrMaxIter'] is None:
        opts['lsqrMaxIter'] = min(200, n)
    if opts['lsmrMaxIter'] is None:
        opts['lsmrMaxIter'] = min(200, n)

    # Initial eval
    r, J = resfun(x)
    r = np.asarray(r).ravel()
    funcCount = 1

    resnorm = r @ r
    f = 0.5 * resnorm
    g = J.T @ r
    gInf = np.max(np.abs(g))
    f0 = f

    diagJtJ = _diag_jtj(J)
    Ddiag = _damping_diag(diagJtJ, opts['damping'])

    mu = opts['mu0'] * max(1.0, np.max(diagJtJ))
    mu = min(max(mu, opts['muMin']), opts['muMax'])
    nu = opts['nu0']

    isJsparse = _is_sparse_J(J, opts['Jstorage'])
    innerMode = _pick_inner_mode(opts, isJsparse)

    lb = _lbfgs_init(opts['lbfgsMem'], n)

    info = {
        'exitflag': 1, 'stopCode': 4,
        'message': 'Maximum iterations reached.',
        'iter': 0, 'funcCount': funcCount,
        'fHist': [f], 'resnormHist': [resnorm], 'gInfHist': [gInf], 'muHist': [mu],
        'rhoHist': [], 'stepNormHist': [], 'innerIterHist': [], 'innerRelresHist': [],
        'acceptedHist': [], 'innerMode': innerMode,
    }

    _display_header(opts['Display'])

    for k in range(1, opts['maxIter'] + 1):
        info['iter'] = k

        if gInf <= opts['tolGrad']:
            info['exitflag'] = 0
            info['stopCode'] = 1
            info['message'] = f"Local minimum: ||g||_inf <= {opts['tolGrad']:.3e}."
            break

        x_base = x
        r_base = r
        J_base = J
        g_base = g
        f_base = f
        diagJtJ_base = diagJtJ
        Ddiag_base = Ddiag
        JtJ = J_base.T @ J_base if innerMode in ('direct', 'direct-aug') else None

        accepted = False

        for t in range(1, opts['maxTrials'] + 1):
            diagA = diagJtJ_base + mu * Ddiag_base

            p, flag, itInner, relresInner = _inner_step(
                innerMode, J_base, r_base, g_base, mu, Ddiag_base, diagA, lb, opts, JtJ)

            if flag != 0 or not np.all(np.isfinite(p)):
                info['exitflag'] = 2
                info['stopCode'] = 5
                info['message'] = f"Inner solve failed (method={innerMode}, flag={flag})."
                break

            stepNorm = np.linalg.norm(p)
            x_try = x_base + p

            res_try = resfun(x_try)
            funcCount += 1
            r_try = np.asarray(res_try[0]).ravel()
            J_try = res_try[1]
            resnorm_try = r_try @ r_try
            f_try = 0.5 * resnorm_try

            if not np.isfinite(f_try):
                rho = -np.inf
            else:
                quad = p @ (JtJ @ p) if JtJ is not None else np.sum((J_base @ p)**2)
                model = f_base + (g_base @ p) + 0.5 * quad + 0.5 * mu * np.sum(Ddiag_base * p**2)
                pred = f_base - model
                act = f_base - f_try
                if not np.isfinite(pred) or pred <= 0:
                    rho = -np.inf
                else:
                    rho = act / pred

            if rho > opts['minAcceptRho'] and np.isfinite(rho):
                accepted = True
                x = x_try
                r = r_try
                J = J_try
                f_old = f_base
                resnorm = resnorm_try
                f = f_try
                g = J.T @ r
                gInf = np.max(np.abs(g))

                # L-BFGS update
                s = x - x_base
                y = g - g_base
                lb = _lbfgs_update(lb, s, y, opts['lbfgsCurvTol'])

                diagJtJ = _diag_jtj(J)
                Ddiag = _damping_diag(diagJtJ, opts['damping'])

                mu = mu * max(1 / 3, 1 - (2 * rho - 1)**3)
                mu = min(max(mu, opts['muMin']), opts['muMax'])
                nu = opts['nu0']

                info['rhoHist'].append(rho)
                info['stepNormHist'].append(stepNorm)
                info['innerIterHist'].append(itInner)
                info['innerRelresHist'].append(relresInner)
                info['acceptedHist'].append(True)

                x_norm = np.linalg.norm(x)
                if stepNorm <= opts['tolStep'] * max(1, x_norm):
                    info['exitflag'] = 0
                    info['stopCode'] = 2
                    info['message'] = f"Relative step <= {opts['tolStep']:.3e}."

                if info['exitflag'] != 0:
                    relChange = abs(f_old - f) / max(1, f0)
                    if relChange <= opts['tolCost']:
                        info['exitflag'] = 0
                        info['stopCode'] = 3
                        info['message'] = f"Relative cost change <= {opts['tolCost']:.3e}."

                break

            else:
                mu = mu * nu
                mu = min(max(mu, opts['muMin']), opts['muMax'])
                nu = min(opts['nuMax'], 2 * nu)
                # Stale curvature degrades preconditioning after rejection
                lb = _lbfgs_init(opts['lbfgsMem'], n)

        if info['exitflag'] == 2:
            break

        if not accepted:
            info['exitflag'] = 3
            info['stopCode'] = 6
            info['message'] = f"No acceptable step after {opts['maxTrials']} trials."
            break

        info['fHist'].append(f)
        info['resnormHist'].append(resnorm)
        info['gInfHist'].append(gInf)
        info['muHist'].append(mu)
        info['funcCount'] = funcCount

        if info['exitflag'] == 0:
            break

    if info['exitflag'] == 1 and info['iter'] >= opts['maxIter']:
        info['message'] = f"Maximum iterations exceeded (maxIter={opts['maxIter']})."

    _display_final(opts['Display'], info)
    return x, info


# --------------- inner step dispatch ---------------
def _inner_step(mode, J, r, g, mu, Ddiag, diagA, lb, opts, JtJ=None):
    n = J.shape[1]
    if mode in ('direct', 'direct-aug'):
        if JtJ is None:
            JtJ = J.T @ J
        A = JtJ.copy()
        A.flat[::n + 1] += mu * Ddiag
        try:
            L = cholesky(A, lower=True)
            p = cho_solve((L, True), -g)
            return p, 0, 0, 0.0
        except np.linalg.LinAlgError:
            # Cholesky failed; try more robust direct solve
            try:
                p = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                return np.zeros(n), 2, 0, 0.0

    elif mode == 'pcg':
        def Afun(v):
            return J.T @ (J @ v) + mu * (Ddiag * v)

        pc = opts['precond']
        if pc == 'none':
            def Mfun(v): return v
        elif pc == 'diag':
            def Mfun(v): return _apply_diag_inv(v, diagA, opts['pcEps'])
        elif pc == 'lbfgs':
            gamma = lb['gamma'] if np.isfinite(lb['gamma']) and lb['gamma'] > 0 else 1.0
            def Mfun(v): return _lbfgs_apply(v, lb, gamma)
        elif pc == 'lbfgs-diag':
            H0inv = 1.0 / (np.maximum(diagA, 0) + opts['pcEps'])
            def Mfun(v): return _lbfgs_apply(v, lb, H0inv)
        else:
            raise ValueError(f"Unknown precond: {pc}")

        return _pcg(Afun, -g, opts['pcgTol'], opts['pcgMaxIter'], Mfun)

    elif mode in ('lsqr', 'lsmr'):
        sqrtMu = np.sqrt(max(mu, 0))
        sqrtD = np.sqrt(np.maximum(Ddiag, 0))
        m = len(r)

        pc = opts['precond']
        if pc == 'none':
            def Pfun(v): return v
        elif pc == 'diag':
            scl = 1.0 / np.sqrt(np.maximum(diagA, 0) + opts['pcEps'])
            def Pfun(v): return scl * v
        elif pc == 'lbfgs':
            gamma = lb['gamma'] if np.isfinite(lb['gamma']) and lb['gamma'] > 0 else 1.0
            def Pfun(v): return _lbfgs_apply(v, lb, gamma)
        elif pc == 'lbfgs-diag':
            H0inv = 1.0 / (np.maximum(diagA, 0) + opts['pcEps'])
            def Pfun(v): return _lbfgs_apply(v, lb, H0inv)
        else:
            raise ValueError(f"Unknown precond: {pc}")

        def Afwd(v):
            Pv = Pfun(v)
            return np.concatenate([J @ Pv, sqrtMu * (sqrtD * Pv)])

        def Atrp(w):
            w1 = w[:m]
            w2 = w[m:]
            return Pfun(J.T @ w1 + sqrtMu * (sqrtD * w2))

        from scipy.sparse.linalg import lsqr, lsmr, LinearOperator
        baug = -np.concatenate([r, np.zeros(n)])

        Aop = LinearOperator((m + n, n), matvec=Afwd, rmatvec=Atrp)

        if mode == 'lsmr':
            result = lsmr(Aop, baug, atol=opts['lsqrTol'], btol=opts['lsqrTol'], maxiter=opts['lsmrMaxIter'])
        else:
            result = lsqr(Aop, baug, atol=opts['lsqrTol'], btol=opts['lsqrTol'], iter_lim=opts['lsqrMaxIter'])

        y = result[0]
        flag = min(result[1], 2)
        iters = result[2]
        p = Pfun(y)
        return p, 0 if flag <= 2 else flag, iters, 0.0

    else:
        raise ValueError(f"Unknown inner mode: {mode}")


def _pcg(Afun, b, tol, maxit, Mfun):
    n = len(b)
    x = np.zeros(n)
    r = b.copy()  # x0=0 so Afun(0)=0, skip the call
    nb = max(1.0, np.linalg.norm(b))
    relres = np.linalg.norm(r) / nb

    if relres <= tol:
        return x, 0, 0, relres

    z = Mfun(r)
    p = z.copy()
    rz_old = r @ z

    for it in range(1, maxit + 1):
        Ap = Afun(p)
        pAp = p @ Ap
        if not np.isfinite(pAp) or pAp <= 0:
            return x, 2, it, relres

        alpha = rz_old / pAp
        x += alpha * p
        r -= alpha * Ap

        relres = np.linalg.norm(r) / nb
        if relres <= tol:
            return x, 0, it, relres

        z = Mfun(r)
        rz_new = r @ z
        if not np.isfinite(rz_new) or rz_new <= 0:
            return x, 2, it, relres

        beta = rz_new / rz_old
        p *= beta
        p += z
        rz_old = rz_new

    return x, 1, maxit, relres


# --------------- utilities ---------------
def _is_sparse_J(J, Jstorage):
    from scipy.sparse import issparse
    if Jstorage == 'auto':
        return issparse(J)
    elif Jstorage == 'sparse':
        return True
    elif Jstorage == 'dense':
        return False
    raise ValueError(f"Unknown Jstorage: {Jstorage}")


def _pick_inner_mode(opts, isJsparse):
    if opts['krylov'] != 'auto':
        return opts['krylov']
    return opts['sparseInner'] if isJsparse else opts['denseInner']




def _apply_diag_inv(u, diagA, pcEps):
    denom = np.maximum(diagA, 0) + pcEps
    return u / denom


# --------------- L-BFGS ---------------
def _lbfgs_init(mem, n):
    """Initialize an L-BFGS history structure.

    Parameters
    ----------
    mem : int  Maximum number of saved corrections.
    n : int    Dimension of the optimization variable.

    Returns
    -------
    dict with keys: mem, S, Y, rho, count, gamma
    """
    return {'mem': mem, 'S': np.zeros((n, mem)), 'Y': np.zeros((n, mem)),
            'rho': np.zeros(mem), 'count': 0, 'gamma': np.nan}


def _lbfgs_update(lb, s, y, curvTol):
    """Push a new {s, y} pair into the L-BFGS history.

    Skips the update if the curvature condition s^T y > curvTol * ||s|| * ||y||
    is not satisfied.

    Parameters
    ----------
    lb : dict      L-BFGS state (mutated in place).
    s : ndarray    Step x_{k+1} - x_k.
    y : ndarray    Gradient difference g_{k+1} - g_k.
    curvTol : float Curvature tolerance.

    Returns
    -------
    dict  The updated L-BFGS state (same object).
    """
    sy = s @ y
    if not np.isfinite(sy):
        return lb
    ns = np.linalg.norm(s)
    ny = np.linalg.norm(y)
    if ns == 0 or ny == 0:
        return lb
    if sy <= curvTol * ns * ny:
        return lb

    if lb['count'] < lb['mem']:
        lb['count'] += 1
        idx = lb['count'] - 1
    else:
        lb['S'][:, :-1] = lb['S'][:, 1:]
        lb['Y'][:, :-1] = lb['Y'][:, 1:]
        lb['rho'][:-1] = lb['rho'][1:]
        idx = lb['mem'] - 1

    lb['S'][:, idx] = s
    lb['Y'][:, idx] = y
    lb['rho'][idx] = 1.0 / sy

    yy = y @ y
    if np.isfinite(yy) and yy > 0:
        lb['gamma'] = sy / yy

    return lb


def _lbfgs_apply(u, lb, H0inv):
    """Apply the two-loop L-BFGS inverse-Hessian approximation to vector u.

    Parameters
    ----------
    u : ndarray (n,)     Vector to multiply.
    lb : dict            L-BFGS state.
    H0inv : float or ndarray  Initial inverse Hessian (scalar or diagonal).

    Returns
    -------
    ndarray (n,)  Approximation of H^{-1} @ u.
    """
    q = np.asarray(u, dtype=float).ravel()
    m = lb['count']

    if m == 0:
        if np.isscalar(H0inv):
            return H0inv * q
        return H0inv * q

    alpha = np.zeros(m)
    for i in range(m - 1, -1, -1):
        si = lb['S'][:, i]
        yi = lb['Y'][:, i]
        alpha[i] = lb['rho'][i] * (si @ q)
        q = q - alpha[i] * yi

    if np.isscalar(H0inv):
        z = H0inv * q
    else:
        z = H0inv * q

    for i in range(m):
        si = lb['S'][:, i]
        yi = lb['Y'][:, i]
        beta = lb['rho'][i] * (yi @ z)
        z = z + si * (alpha[i] - beta)

    return z


# --------------- defaults + display ---------------
_DEFAULTS = {
    'maxIter': 100, 'maxTrials': 10,
    'tolGrad': 1e-8, 'tolStep': 1e-10, 'tolCost': 1e-12,
    'mu0': 1e-2, 'nu0': 2, 'muMin': 1e-15, 'muMax': 1e15, 'nuMax': 1e15,
    'damping': 'diagJTJ', 'minAcceptRho': 1e-3,
    'Jstorage': 'auto', 'krylov': 'auto',
    'sparseInner': 'direct', 'denseInner': 'direct-aug',
    'pcgTol': 1e-3, 'pcgMaxIter': None,
    'lsqrTol': 1e-3, 'lsqrMaxIter': None, 'lsmrMaxIter': None,
    'precond': 'lbfgs-diag', 'pcEps': 1e-12,
    'lbfgsMem': 10, 'lbfgsCurvTol': 1e-12,
    'Display': 'off',
}


def _display_header(Display):
    if Display in ('off', 'final'):
        return
    if Display == 'iter':
        print(f"\n{'Iteration':>10} {'Func-count':>10} {'Resnorm':>14} {'optimality':>14} {'Lambda':>12} {'step':>12}")
        print('-' * 74)
    elif Display == 'iter-detailed':
        headers = ['Iteration', 'Func-count', 'Resnorm', 'optimality', 'Lambda', 'step', 'rho', 'inIt', 'inRelres', 'Status']
        print("\n" + "".join(f"{h:>12}" for h in headers))


def _display_final(Display, info):
    if Display == 'off':
        return
    if Display == 'final':
        print(f"\n{info['message']}")
    elif Display in ('iter', 'iter-detailed'):
        f_last = info['fHist'][-1] if info['fHist'] else float('nan')
        print(f"\nStop reason: {info['message']}")
        print(f"exitflag={info['exitflag']}, stopCode={info['stopCode']}, "
              f"funcCount={info['funcCount']}, f={f_last:.6e}")

