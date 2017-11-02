from collections import OrderedDict
from dolfin import plot, norm
import numpy as np
import matplotlib.pyplot as pl


def running(x, N=5):
    return np.convolve(x, np.ones((N,))/N, mode='valid')


def get_longest(res:dict, theta:float):
    ll = [v for k, v in res.items() if v['theta'] == theta]
    maxsteps = 0
    run = {}
    for r in ll:
        if r['steps'] > maxsteps:
            run = r
            maxsteps = r['steps']
    return run


def plots1(history:dict, _slice=slice(0,-1), running_mean_window=1):
    h = history
    pl.figure(figsize=(18,12), )
    pl.suptitle("'%s', $\\theta = %.2e$, $\mu = %.2e$, $\\epsilon = %.2e$"
                % (h['init'], h['theta'], h['mu'], h['e_stop']))
    pl.subplot(3,2,1)
    pl.plot(running(h['du'][_slice], running_mean_window))
    pl.title('$d_{t}u$, window: %d' % running_mean_window)
    pl.subplot(3,2,2)
    pl.plot(running(h['dv'][_slice], running_mean_window))
    pl.title('$d_{t}v$, window: %d' % running_mean_window)
    pl.subplot(3,2,3)
    pl.plot(running(np.log(h['alpha'][_slice]), running_mean_window))
    pl.title('$log\ \\alpha_t$, window: %d' % running_mean_window)
    pl.subplot(3,2,4)
    pl.plot(h['constraint'][_slice])
    pl.title("constraint")
    pl.subplot(3,2,5)
    pl.plot(h['symmetry'][_slice])
    pl.title("symmetry")
    pl.subplot(3,2,6)
    pl.plot(h['J'][_slice])
    pl.title("Energy")


def plots2(history:dict):
    h = history
    pl.figure(figsize=(18,18))
    pl.subplot(2,2,1)
    plot(h['u'], title="$u_{\\theta}$ at last timestep, norm = %.2e" % norm(h['u']))
    pl.subplot(2,2,2)
    plot(h['v'], title="$v_{\\theta}$ at last timestep, norm = %.2e" % norm(h['v']))
    pl.subplot(2,2,3)
    plot(h['dtu'], title="$du_{\\theta}$ at last timestep, norm = %.2e" % norm(h['dtu']))
    pl.subplot(2,2,4)
    plot(h['dtv'], title="$dv_{\\theta}$ at last timestep, norm = %.2e" % norm(h['dtv']))


def plots3(run: dict, begin: float = 0.0, end: float = np.inf):
    frun = {k: v for k, v in run.items() if begin <= v['theta'] < end}
    r = OrderedDict(sorted(frun.items(), key=lambda x: x[1]['theta']))

    thetas = [r['theta'] for k, r in r.items()]
    syms = [1. / np.array(r['symmetry'][-4:]).mean() for k, r in r.items()]
    JJ = [np.array(r['J'][-4:]).mean() for k, r in r.items()]
    J0 = [r['J'][0] for k, r in r.items()]
    steps = [r['steps'] for k, r in r.items()]
    fig = pl.figure(figsize=(14, 14))
    ax = pl.subplot(2, 1, 1)
    pl.plot(thetas, syms, 'o', thetas, syms, 'k')
    for xy, step in zip(zip(thetas, syms), steps):
        ax.annotate('%d' % step, xy=np.array(xy) * np.array([1.0, 1.0001]), textcoords='data')
    pl.xticks(thetas, thetas)
    pl.xlabel("$\\theta$")
    pl.ylabel("Symmetry")
    pl.title("Symmetry of the solution as a function of $\\theta$")
    # pl.subplot(2,2,2)
    # pl.plot(thetas, steps)
    # pl.xticks(thetas, thetas)
    # pl.xlabel("$\\theta$")
    # pl.ylabel("Number of steps taken")
    # pl.title("Iterations to convergence")
    pl.subplot(2, 2, 3)
    pl.plot(thetas, J0)
    pl.xticks(thetas, thetas)
    pl.title("Initial energy")
    pl.xlabel("$\\theta$")
    pl.ylabel("J")
    pl.subplot(2, 2, 4)
    pl.plot(thetas, JJ)
    pl.xticks(thetas, thetas)
    pl.title("Final energy")
    pl.xlabel("$\\theta$")
    pl.ylabel("J")


def plots4(runs, _slice=slice(0, -1), running_mean_window=1):
    if isinstance(runs, filter):
        _runs = [v for k, v in sorted(runs, key=lambda x: x[1]['theta'])]
    elif isinstance(runs, dict):
        _runs = [v for k, v in sorted(runs.items(), key=lambda x: x[1]['theta'])]
    pl.figure(figsize=(18, 12), )
    pl.suptitle("'%s'" % _runs[0]['init'])
    pl.subplot(3, 2, 1)
    for h in _runs:
        pl.plot(running(h['du'][_slice], running_mean_window), label='$\\theta = %.2f$' % h['theta'])
    pl.title('$d_{t}u$, window: %d' % running_mean_window)
    pl.legend()
    pl.subplot(3, 2, 2)
    for h in _runs:
        pl.plot(running(h['dv'][_slice], running_mean_window), label='$\\theta = %.2f$' % h['theta'])
    pl.title('$d_{t}v$, window: %d' % running_mean_window)
    pl.legend()
    pl.subplot(3, 2, 3)
    for h in _runs:
        pl.plot(running(np.log(h['alpha'][_slice]), running_mean_window), label='$\\theta = %.2f$' % h['theta'])
    pl.title('$log\ \\alpha_t$, window: %d' % running_mean_window)
    pl.legend()
    pl.subplot(3, 2, 4)
    for h in _runs:
        pl.plot(h['constraint'][_slice], label='$\\theta = %.2f$' % h['theta'])
    pl.title("constraint")
    pl.legend()
    pl.subplot(3, 2, 5)
    xmax = 0
    for h in _runs:
        xmax = max(xmax, len(h['symmetry'][_slice]))
        pl.plot(h['symmetry'][_slice], label='$\\theta = %.2f$' % h['theta'])
    pl.hlines(_runs[0]['symmetry'][0], xmin=0, xmax=xmax, linestyles='dotted')
    pl.title("symmetry")
    pl.legend()
    pl.subplot(3, 2, 6)
    for h in _runs:
        pl.plot(h['J'][_slice], label='$\\theta = %.2f$' % h['theta'])
    pl.title("Energy")
    pl.legend()

