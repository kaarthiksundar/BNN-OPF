from functools import partial 
import jax.experimental.sparse as sparse
import jax

@partial(jax.jit, static_argnums=(2))
def sd_matmul(A, B, shape):
    """ 
    Arguments: 
        A: (nxm) sparse matrix 
        B: (mxk) dense matrix 
        shape: value of n 
    Returns: 
        (nxk) dense matrix
    """
    indices = A.indices
    values = A.data 
    rows, cols = indices.transpose()
    in_ = B.take(cols, axis=0)
    prod = in_*values[:, None]
    res = jax.ops.segment_sum(prod, rows, shape)
    return res

