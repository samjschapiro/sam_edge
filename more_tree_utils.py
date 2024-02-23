# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Misc. utilities for pytrees."""
import jax
from jax import jit
from jax import tree_util
import jax.numpy as jnp
import hessian_norm as hes

NORMALIZING_EPS = 1e-5
NORMALIZING_EPS_KL = 1e-9


@jit
def normalize(t):
  norm = get_vector_norm(t)
  return tree_util.tree_map(lambda t_leaf: t_leaf/(norm + NORMALIZING_EPS), t)

@jit
def normalize_sum(t):
  norm = get_vector_sum(t)
  return tree_util.tree_map(lambda t_leaf: t_leaf/(norm + NORMALIZING_EPS), t)

@jit
def project_out_and_normalize(s, t):
  """Project out and normalize.

  Args:
    s: pytree
    t: another pytree

  Returns:
    The pytree obtained by projecting the flattening of
    t onto the flattening of s, normalizing the result,
    and then reshaping into a pytree.
  """

  s_dot_t = get_tree_dot(s, t)
  # pylint: disable=g-long-lambda
  new_part = tree_util.tree_map(lambda s_leaf, t_leaf:
                                t_leaf - s_dot_t * s_leaf, s, t)
  return normalize(new_part)


def get_orthonormal_basis(t_list):
  k = len(t_list)
  t_list[0] = normalize(t_list[0])
  for i in range(k):
    for j in range(i+1, k):
      t_list[j] = project_out_and_normalize(t_list[i], t_list[j])
  return t_list


@jit
def get_vector_norm(t):
  squared_norms = tree_util.tree_map(lambda x: jnp.sum(x*x), t)
  return jnp.sqrt(jnp.sum(jnp.array(tree_util.tree_leaves(squared_norms))))

@jit
def get_vector_sum(t):
  sums = tree_util.tree_map(lambda x: jnp.sum(x), t)
  return jnp.sum(jnp.array(tree_util.tree_leaves(sums)))

def get_vector_unif_kl(t, num_params):
  normed_t = normalize_sum(tree_util.tree_map(lambda x: jnp.abs(x), t))
  entropies = tree_util.tree_map(lambda x: jnp.sum(x*jnp.log((NORMALIZING_EPS_KL+x)*num_params)), normed_t)
  return jnp.average(jnp.array(tree_util.tree_leaves(entropies)))

# @jit
# def get_rank_of_second_moment(t):
#   # note: wayyyy too slow
#   second_moment_ranks = tree_util.tree_map(lambda x: jnp.linalg.matrix_rank(jnp.outer(x, x)), t)
#   return jnp.linalg.matrix_rank(jnp.array(tree_util.tree_leaves(second_moment_ranks)))

# @jit
# def get_vector_l1_norm(t):
#   num_greater_than_threshold = tree_util.tree_map(lambda x: jnp.linalg.norm(x, ord=1), t)
#   return jnp.mean(jnp.array(tree_util.tree_leaves(num_greater_than_threshold)))

@jit
def get_absolute_component_skew(t, thresh):
  normed_t = normalize(t)
  absolute_t = tree_util.tree_map(lambda x: jnp.count_nonzero((jnp.abs(x) >= thresh).astype(int)), normed_t)
  return jnp.mean(jnp.array(tree_util.tree_leaves(absolute_t)))

def count_parameters(t):
  leaf_parameter_counts = tree_util.tree_map(lambda x: x.size, t)
  return jnp.sum(jnp.array(tree_util.tree_leaves(leaf_parameter_counts)))


@jit
def get_tree_dot(s, t):
  leaf_dots = tree_util.tree_map(lambda si, ti: jnp.sum(si*ti), s, t)
  return jnp.sum(jnp.array(tree_util.tree_leaves(leaf_dots)))


@jit
def get_alignment(s, t):
  return jnp.abs(get_tree_dot(s, t))/(get_vector_norm(s)*get_vector_norm(t))


@jit
def get_random_direction(rng_key, t):
  """Sample a unit length pytree.

  Args:
    rng_key: RNG key
    t: a pytree

  Returns:
    A pytree with the same shape as t, whose
    leaves are collectively sampled uniformly
    at random from the unit ball.
  """
  def sample_at_leaf(sub_key, shape):
    return jax.random.normal(sub_key, shape)

  flat_t, treedef = tree_util.tree_flatten(t)
  leaf_shapes = tree_util.tree_map(lambda x: x.shape, flat_t)
  rng_keys = jax.random.split(rng_key, len(leaf_shapes))

  new_leaves = [sample_at_leaf(rng_keys[i], leaf_shapes[i])
                for i in range(len(leaf_shapes))]
  new_leaves = normalize(new_leaves)
  return tree_util.tree_unflatten(treedef, new_leaves)
