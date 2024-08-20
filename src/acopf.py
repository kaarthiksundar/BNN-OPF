import jax
import jax.numpy as jnp
import numpy as np

# AC-OPF helper functions
# get input variables into individual parts
def get_input_variables(X, opf_data):
    nl = opf_data.get_num_loads()
    pd = X[:, :nl]
    qd = X[:, -nl:]
    return pd, qd

# split output prediction to individual parts
def get_output_variables(Y, opf_data):
    ng = opf_data.get_num_gens()
    nbus = opf_data.get_num_buses()
    pg = Y[:, :ng]
    qg = Y[:, ng:2*ng]
    vm = Y[:, -2*nbus:-nbus]
    va = Y[:, -nbus:]
    return pg, qg, vm, va

# evaluate objective function given output predictions 
def get_objective_value(Y, opf_data) -> jax.Array:
    pg, _, _, _ = get_output_variables(Y, opf_data)
    cost = (opf_data.gen_cost.q * pg**2).sum(axis=1) + (opf_data.gen_cost.l * pg).sum(axis=1) + opf_data.gen_cost.c 
    return cost

# evaluate equality constraint residuals given input and output data (PF constraints)
def get_equality_constraint_violations(X, Y, opf_data) -> jax.Array: 
    pg, qg, vm, va = get_output_variables(Y, opf_data)
    pd, qd = get_input_variables(X, opf_data)
    # voltage shape: (num_samples * num_buses)
    voltage = vm * np.cos(va) + 1j * vm * np.sin(va)
    bus_injection = np.multiply(voltage, np.conjugate(voltage * np.transpose(opf_data.y_bus)))
    generation = jnp.zeros(
        (pg.shape[0], opf_data.get_num_buses()), dtype=complex
        ).at[:, opf_data.gen_bus_idx].set(pg + 1j * qg)
    load = jnp.zeros(
        (pd.shape[0], opf_data.get_num_buses()), dtype=complex
        ).at[:, opf_data.load_bus_idx].set(pd + 1j * qd)
    residual = generation - load - bus_injection 
    return jnp.concatenate([np.real(residual), np.imag(residual)], axis=1)

# evaluate inequality constraint residuals given input and output data (variable bounds)
def get_inequality_constraint_violations(Y, opf_data, line_limits = False) -> jax.Array:
    pg, qg, vm, va = get_output_variables(Y, opf_data)
    pg_lower = jnp.maximum(opf_data.pg_bounds.lower - pg, 0.0)
    pg_upper = jnp.maximum(pg - opf_data.pg_bounds.upper, 0.0)
    qg_lower = jnp.maximum(opf_data.qg_bounds.lower - qg, 0.0)
    qg_upper = jnp.maximum(qg - opf_data.qg_bounds.upper, 0.0)
    vm_lower = jnp.maximum(opf_data.vm_bounds.lower - vm, 0.0)
    vm_upper = jnp.maximum(vm - opf_data.vm_bounds.upper, 0.0)
    if (line_limits == False):
        residual = jnp.concatenate([
            pg_lower, pg_upper, 
            qg_lower, qg_upper, 
            vm_lower, vm_upper
        ], axis=1)
        return residual
    # compute branch flow (do calculation later)
    branch_flow = jnp.array([
        np.multiply(
            jnp.array([
            vm[y_branch.idx[0]] * (
                np.cos(va[y_branch.idx[0]]) + 1j * np.sin(va[y_branch.idx[0]])
                ), 
            vm[y_branch.idx[1]] * (
                np.cos(va[y_branch.idx[1]]) + 1j * np.sin(va[y_branch.idx[1]])
                )
            ]), 
            np.conjugate(
                jnp.array([
                    vm[y_branch.idx[0]] * (
                        np.cos(va[y_branch.idx[0]]) + 1j * np.sin(va[y_branch.idx[0]])
                        ), 
                    vm[y_branch.idx[1]] * (
                        np.cos(va[y_branch.idx[1]]) + 1j * np.sin(va[y_branch.idx[1]])
                        )
                    ]), 
                np.transpose(y_branch.admittance_matrix)    
            )  
        ) 
        for y_branch in opf_data.y_branch
    ])
    pass

# Physics-driven Loss function evaluation, X Y should be un-normalised data 
def assess_feasibility(X, Y, opf_data, eq_weight = 1.0, ineq_weight = 1.0):
    eq = get_equality_constraint_violations(X, Y, opf_data)
    ineq = get_inequality_constraint_violations(Y, opf_data)
    return eq_weight * (eq**2).sum(axis=1) + ineq_weight * (ineq**2).sum(axis=1)
    