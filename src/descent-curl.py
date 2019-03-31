# coding: utf-8

# # Problem definition
# 
# We wish to minimize
#
# $$ I(u,v) = \frac{\theta}{2} \int_{\omega} |\nabla_s u +
#             \tfrac{1}{2} \nabla v \otimes \nabla v|^{2} \mathrm{d}x
#             + \frac{1}{24} \int_{\omega} |\nabla^2 v -
#             \mathrm{Id}|^{2} \mathrm{d}x. $$
# 
# Because we only have $C^0$ elements we set $z$ for $\nabla v$ and
# minimize instead
# 
# $$ J(u,z) = \frac{\theta}{2} \int_{\omega} |\nabla_s u +
#             \tfrac{1}{2} z \otimes z|^{2} \mathrm{d}x + \frac{1}{24}
#             \int_{\omega} |\nabla z - \mathrm{Id}|^{2} \mathrm{d}x +
#             \mu \int_{\omega} |\mathrm{curl}\ z|^{2} \mathrm{d}x, $$
# 
# then recover the vertical displacements (up to a constant) by
# minimizing
# 
# $$ F(p,q) = \tfrac{1}{2} || \nabla p - q ||^2 + \tfrac{1}{2} || q -
#             z ||^2. $$
# 
# This we do by solving the linear problem $D F = 0$.
# 
# Minimization of the energy functional $J$ is done via gradient
# descent and a line search. In plane displacements and gradients of
# out of plane displacements form a mixed function space $U \times
# Z$. We also have another scalar space $V$ where the potential of the
# out of plane gradients lives. The model is defined and solved in
# `run_model()` below. Experiments can be easily run in parallel with
# `joblib`.

from dolfin import *
import numpy as np
from tqdm import tqdm
from common import *
from time import time


def noop(*args, **kwargs):
    pass


def run_model(init: str, qform: str, mesh_file: str, theta: float, mu_scale:
              float = 1.0, dirichlet_size: int = -1, deg: int = 1,
              e_stop_mult: float = 1e-5, max_steps: int = 1000, skip:
              int = 10, save_funs: bool = True, debug=print, n=0):
    """
    Parameters
    ----------
        init: Initial condition. One of 'zero', 'rot', 'parab', 'ani_parab',
              'iso_compression', 'ani_compression', 'bartels'
        qform: Quadratic form to use: 'frobenius' or 'isotropic' (misnomer...)
        mesh_file: name of (gzipped) xml file with the mesh data
        theta: coefficient for the nonlinear in-/out-of-plane mix of stresses
        mu_scale: compute penalty weight as mu_scale / msh.hmin()
        dirichlet_size: -1 to deactivate Dirichlet BCs, 0 for one cell.
                        > 0 to recursively enlarge the Dirichlet domain.
        deg: polynomial degree to use
        e_stop_mult: Multiplier for the stopping condition.
        max_steps: Fallback maximum number of steps for gradient descent.
        skip: save displacements every so many steps.
        save_funs: Whether to store the last values of the solutions and updates
                   in the returned dictionary (useful for plotting in a notebook but
                   useless for pickling)
        debug: set to noop or print
        n: index of run in a parallel computation for the displaying of progress bars
    """
    set_log_level(ERROR)
    
    impl = 'curl-proj-dirichlet' if dirichlet_size >= 0 else 'curl-proj'
    t = tqdm(total=max_steps, desc='th=% 8.3f' % theta, position=n, dynamic_ncols=True)

    MARKER = 1
    msh = Mesh(mesh_file)
    subdomain = FacetFunction("uint", msh, 0)
    recursively_intersect(msh, subdomain, Point(0, 0), MARKER, recurr=dirichlet_size)
    mu = mu_scale / msh.hmin()

    # FIXME: generalise symmetry calculations to avoid this hack
    # Even better, use curvature instead
    if mesh_file.lower().find('circle') >= 0:
        symmetry = circular_symmetry
    elif mesh_file.lower().find('rectangle') >= 0:
        symmetry = rectangular_symmetry
    else:
        raise ValueError("Unsupported mesh geometry for symmetry calculation")
    
    # In-plane displacements (IPD)
    UE = VectorElement("Lagrange", msh.ufl_cell(), deg, dim=2)
    U = FunctionSpace(msh, UE)

    # Gradients of out-of-plane displacements (OPD)
    ZE = VectorElement("Lagrange", msh.ufl_cell(), deg, dim=2)
    Z = FunctionSpace(msh, ZE)
    
    # Mixed function space u,z
    W = FunctionSpace(msh, UE * ZE)

    # For the curvature computation
    T = TensorFunctionSpace(msh, 'DG', 0)
    
    # Removing the antisymmetric part of the gradient requires constructing
    # functions in subspaces of W, which does not work because of dof orderings
    # A solution is to collapse() the subspaces, but again dof ordering is not
    # kept. In the end I'll just copy stuff around until I find something better.
    # HACK HACK HACK, inefficient, this sucks
    fau = FunctionAssigner(W.sub(0), U)
    faz = FunctionAssigner(W.sub(1), Z)
    rfau = FunctionAssigner(U, W.sub(0))
    rfaz = FunctionAssigner(Z, W.sub(1))
    
    # will store out-of-plane displacements (potential of z)
    V = FunctionSpace(msh, "Lagrange", deg)

    # We gather in-plane and out-of-plane displacements into one
    # Function for visualization with ParaView.
    P = VectorFunctionSpace(msh, "Lagrange", deg, dim=3)
    faux = FunctionAssigner(P.sub(0), W.sub(0).sub(0))
    fauy = FunctionAssigner(P.sub(1), W.sub(0).sub(1))
    fav = FunctionAssigner(P.sub(2), V)
    disp = Function(P)
    disp.rename("disp", "displacement")

    qform = qform.lower()
    file_name = make_filename(impl, init, qform, theta, mu)
    file = File(file_name, "compressed")  # .vtu files will have the same prefix

    def save_displacements(u, z, step):
        debug("\tSaving... ", end='')
        v = compute_potential(z, V, subdomain, MARKER, 0.0)
        faux.assign(disp.sub(0), u.sub(0))
        fauy.assign(disp.sub(1), u.sub(1))
        fav.assign(disp.sub(2), v)
        file << (disp, float(step))        
        debug("Done.")
        
    
    bcW = DirichletBC(W, Constant((0.0, 0.0, 0.0, 0.0)), subdomain, MARKER)

    # Solution at time t ("current step")
    w = Function(W)
    u, z = w.split()
    # Solution at time t-1 ("previous step")
    w_ = Function(W)
    u_, z_ = w_.split()
    # Gradient representative in the FE space
    dw = Function(W)
    du, dz = dw.split()
    
    # Initial condition
    w_init = make_initial_data_penalty(init)
    w.interpolate(w_init)
    w_.interpolate(w_init)
    save_displacements(u, z, 0)  # Output it too
    
    # Setup forms and energy
    if qform == 'frobenius':
        Q2, L2 = frobenius_form()
    elif qform == 'isotropic':
        # Isotropic density for steel at room temp.
        # http://scienceworld.wolfram.com/physics/LameConstants.html
        # breaks things (line searches don't end) because we need to scale
        # elastic constants with h
        E, nu = 210.0, 0.3
        Q2, L2 = isotropic_form(E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 + 2 * nu))
    else:
        raise Exception("Unknown quadratic form name '%s'" % qform)

    B = Identity(2)
    zero_energy = assemble((1. / 24) * inner(B, B) * dx(msh))

    def energy(u, z, mu=mu):
        J = (theta / 2.) * Q2(eps(u) + outer(z, z) / 2) * dx(msh) \
            + (1. / 24) * Q2(grad(z) - B) * dx(msh) \
            + mu * inner(curl(z), curl(z)) * dx(msh)
        return assemble(J)
    
    cur_energy = energy(u, z)
    
    ####### Set up gradient descent method and history
    
    e_stop = msh.hmin() * e_stop_mult
    max_line_search_steps = 20
    fail = False
    step = 0
    omega = 0.25  # Gradient descent fudge factor in (0, 1/2)
    alpha = ndu = ndz = 1.0
    
    _hist = {'init': init, 'impl': impl, 'deg': deg, 'mesh': mesh_file,
             'dirichlet': dirichlet_size, 'mu': mu, 'theta': theta, 'e_stop': e_stop,
             'J': [], 'alpha': [], 'du': [], 'dz': [], 'constraint': [],
             'Q2': {'form_name': Q2.__name__, 'arguments': Q2.arguments},
             'Kxx': [], 'Kxy': [], 'Kyy': [],'symmetry': [], 'file_name': file_name}
    
    debug("Solving with theta = %.2e, mu = %.2e, eps=%.2e for at most %d steps."
          % (theta, mu, e_stop, max_steps))
    
    # LHS for the gradient computation
    # Careful!! Picking the right scalar product here is essential
    # Recall the issues with boundary values: integrate partially 
    # and only boundary terms survive...
    dtu, dtz = TrialFunctions(W)
    phi, psi = TestFunctions(W)
    L = inner(dtu, phi) * dx + inner(grad(dtu), grad(phi)) * dx \
        + inner(dtz, psi) * dx + inner(grad(dtz), grad(psi)) * dx

    begin = time()
    domain_area = assemble(1*dx(msh))
    
    while alpha * (ndu ** 2 + ndz ** 2) > e_stop and step < max_steps and not fail:
        _curl = assemble(curl(z_) * dx)
        K = project(sym(grad(z_)), T)
        _hist['Kxx'].append(assemble(K[0,0]*dx)/domain_area)
        _hist['Kxy'].append(assemble(K[0,1]*dx)/domain_area)
        _hist['Kyy'].append(assemble(K[1,1]*dx)/domain_area)
        
        _symmetry = symmetry(disp)
        _hist['constraint'].append(_curl)
        _hist['symmetry'].append(_symmetry)
        debug("Step %d, energy = %.3e, curl = %.3e, symmetry = %.3f"
              % (step, cur_energy, _curl, _symmetry))

        #### Gradient
        # for some reason I'm not able to use derivative(J, w_, dtw)
        dJ = theta * L2(eps(u_) + outer(z_, z_) / 2,
                        eps(phi) + sym(outer(z_, psi))) * dx(msh) \
             + (1. / 12) * L2(grad(z_) - B, grad(psi)) * dx(msh) \
             + 2 * mu * inner(curl(z_), curl(psi)) * dx(msh)
        debug("\tSolving...", end='')
        # Since u_, v_ are given from the previous iteration, the
        # problem is linear in phi,psi.
        solve(L == -dJ, dw, [bcW])
        
        # dw is never reassigned to a new object so it should be ok
        # to reuse du, dv without resplitting right?
        du, dz = dw.split()
        
        ndu = norm(du)
        ndz = norm(dz)

        debug(" done with |du| = %.3f, |dz| = %.3f" % (ndu, ndz))

        #### Line search
        new_energy = 0
        debug("\tSearching... ", end='')
        while not fail:
            w = project(w_ + alpha * dw, W)
            u, z = w.split()
            nu = norm(u)
            nz = norm(z)
            debug(" new u,z with |u| = %.3f, |z| = %.3f, alpha=%.4f" % (nu, nz, alpha))
            new_energy = energy(u, z)
            if new_energy <= cur_energy - omega * alpha * (ndu ** 2 + ndz ** 2):
                debug(" alpha = %.2e" % alpha)
                _hist['J'].append(cur_energy)
                _hist['alpha'].append(alpha)
                _hist['du'].append(ndu)
                _hist['dz'].append(ndz)
                cur_energy = new_energy
                alpha = min(1.0, 2.0 * alpha)  # Use a larger alpha for the next line search
                break
            if alpha < (1. / 2) ** max_line_search_steps:
                fail = True
                debug("Line search failed after %d steps" % max_line_search_steps)
            alpha /= 2.0  # Repeat with smaller alpha
        step += 1
        
        # Project onto space of admissible displacements
        u, z = Function(U), Function(Z)
        rfau.assign(u, w.sub(0))
        rfaz.assign(z, w.sub(1))
        #center_function(u, dim=2)
        #center_function(z, dim=2)
        symmetrise_gradient(u, U)
        fau.assign(w.sub(0), u)
        faz.assign(w.sub(1), z)
        
        # HACK: go back to functions over subspaces
        u, z = w.split()
        w_.vector()[:] = w.vector()
        u_, z_ = w_.split()

        if step % skip == 0:
            save_displacements(u, z, step)
        t.update()

    _hist['time'] = time() - begin

    if step < max_steps:
        t.total = step
        t.update()

    _hist['steps'] = step
    if save_funs:
        _hist['disp'] = disp
        _hist['u'] = u
        _hist['v'] = compute_potential(z, V, subdomain, MARKER, 0.0)
        _hist['dtu'] = du
        _hist['dtz'] = dz
    debug("Done after %d steps" % step)

    t.close()
    return _hist


if __name__ == "__main__":

    from joblib import Parallel, delayed

    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["cpp_optimize"] = True

    results_file = "results-combined.pickle"
    # mesh_file = generate_mesh('circle', 18, 18)
    # theta_values = [150, 160, 170, 180, 190, 200, 300, 400, 500, 700, 900, 1100]
    mesh_file = generate_mesh('circle', 36, 36)
    theta_values = [200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1500]

    
    # Careful: hyperthreading won't help (we are probably bound by memory channel bandwidth)
    n_jobs = min(12, len(theta_values))

    new_res = Parallel(n_jobs=n_jobs)(delayed(run_model)('ani_parab', 'frobenius', mesh_file,
                                                         theta=theta, mu=theta/10.0,
                                                         dirichlet_size=0, deg=1,
                                                         max_steps=15000, save_funs=False, skip=10,
                                                         e_stop_mult=1e-8, debug=noop, n=n)
                                      for n, theta in enumerate(theta_values))

    save_results(new_res, results_file)
