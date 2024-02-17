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

"""Perform experiments regarding SAM and the edge of stability."""
import dataclasses
import math
import time

import hessian_norm
from jax import grad
from jax import jit
from jax import tree_util
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import more_tree_utils as mtu

EPSILON = 1e-5
DPI = 300
DIFFERENTIATION_STEP_SIZE = 0.001
POWER_ITERATION_EPS = 0.001  # relative error

# pylint: disable=anomalous-backslash-in-string


def train(params,
          model,
          loss,
          train_batches,
          step_size,
          rho,  # for SAM -- if rho is 0.0, SAM is not used
          hessian_check_gap,
          eigs_curve_filename,
          eigs_se_only_filename,
          alignment_curve_filename,
          loss_curve_filename,
          raw_data_filename,
          num_principal_comps,
          time_limit_in_hours,
          second_order,
          rng):
  """Train a model using SAM, and plot statistics.

  Args:
    params: parameters of the model
    model: the model
    loss: the loss function
    train_batches: training data
    step_size: learning rate
    rho: distance uphill to evaluate the gradient
    hessian_check_gap: time in seconds hessian evaluation
    eigs_curve_filename: name of PDF file for eigenvalue/edge plots
    eigs_se_only_filename: name of PDF file for eigenvalue/edge plots
            without 2/eta
    alignment_curve_filename: name of PDF file for plots of alignments
    loss_curve_filename: name of PDF file for loss curves
    raw_data_filename: name of file for raw data
    num_principal_comps: number of the principal components of the
            hessian to evaluate
    time_limit_in_hours: Time limit
    rng: key

  Returns:
    final parameters
  """

  @jit
  def loss_by_params(params, x_batched, y_batched):
    preds = model.apply(params, x_batched)
    return jnp.mean(loss(x_batched, preds, y_batched))

  @jit
  def sam_neighbor(params, x, y):
    grads = grad(loss_by_params)(params, x, y)
    norm = mtu.get_vector_norm(grads)
    return tree_util.tree_map(lambda p, g: p + rho * g/(norm + EPSILON),
                              params,
                              grads)

  # @jit
  # def get_max_eigenvector(params, x, y):
  #   max_eig_vecs = tree_util.tree_map(lambda p, x, y: ce.curvature_and_direction(p, x, y)[1], params, x, y)  
  #   max_eig_vecs_norms = mtu.get_vector_norm(max_eig_vecs)
  #   return max_eig_vecs, max_eig_vecs_norms
  
  @jit
  def second_order_sam_neighbor(params, x, y, approx_hessian=False):
    if not approx_hessian:
      v1,_ = tree_util.tree_map(lambda p, x, y: ce.curvature_and_direction(p, x, y)[1], params, x, y)
      norm = mtu.get_vector_norm(v1)
      return tree_util.tree_map(lambda p: p + rho * v1/(norm + EPSILON), params)

  @jit
  def update(params, x, y, eta):
    if rho > 0.0:
      grad_location = sam_neighbor(params, x, y) if second_order == False else second_order_sam_neighbor(params, x, y)
    else:
      grad_location = params
    grads = grad(loss_by_params)(grad_location, x, y)
    return tree_util.tree_map(lambda p, g: p - eta * g,
                              params,
                              grads)

  @jit
  def get_sam_gradient(params, x, y):
    grad_location = sam_neighbor(params, x, y)
    return grad(loss_by_params)(grad_location, x, y)

  @jit
  def get_second_order_sam_gradient(params, x, y):
    grad_location = second_order_sam_neighbor(params, x, y)
    return grad(loss_by_params)(grad_location, x, y)
  
  # NEW!! #TODO: JIT THIS!

  @jit
  def hessian_vector_product(params, x, y, vector):
    """Compute the product between the Hessian and a vector.

  Args:
    params: a pytree with the parameters of the model
    x: feature tensor
    y: label tensor
    vector: a pytree of the same shape as params

  Returns:
    The product of the Hessian of the loss with the vector.
  """
    eps = DIFFERENTIATION_STEP_SIZE

    plus_location = tree_util.tree_map(lambda theta, v: theta + eps*v,
                                        params,
                                        vector)
    plus_gradient = grad(loss_by_params)(plus_location, x, y)
    minus_location = tree_util.tree_map(lambda theta, v: theta - eps*v,
                                        params,
                                        vector)
    minus_gradient = grad(loss_by_params)(minus_location, x, y)
    # pylint: disable=g-long-lambda
    return tree_util.tree_map(lambda x, y: ((x - y)/
                                            (2.0*eps
                                              + mtu.NORMALIZING_EPS)),
                              plus_gradient,
                              minus_gradient)

  def curvature_and_direction(self, params, x, y):
      self.rng, subkey = jax.random.split(self.rng)
      v = mtu.get_random_direction(subkey, params)

      return self.curvature_and_direction_with_start(params, x, y, v)
  
  @jit
  def iteration(old_v):
    # perform hessian-vector product
    product = hessian_vector_product(params, x, y, old_v)

    # calculate the norm
    product_norm = mtu.get_vector_norm(product)

    # normalize the product to get a new iterate v
    new_v = jax.tree_util.tree_map(lambda w: w/product_norm, product)

    return new_v, product, product_norm

  @jit # TODO: not sure about this! double check
  def curvature_and_direction_with_start(self, params, x, y, v):
    """Compute the curvature of the loss with respect to x and y.

    Args:
      params: a pytree with the parameters of the model
      x: feature tensor
      y: label tensor
      v: a pytree of the same shape as params, to use as a warm start

    Returns:
      The principal eigenvalue of the Hessian (which could be negative),
      and the principal eigenvector.
    """


    last_norm = None
    current_norm = 1.0
    t = 0
    while (((last_norm is None)
            or ((abs(current_norm - last_norm)
                > current_norm*POWER_ITERATION_EPS)
                and abs(current_norm) > POWER_ITERATION_EPS))
          and (t < self.iteration_limit)):
      last_norm = current_norm
      v, product, current_norm = iteration(v)
      t += 1

    leaf_dots_tree = tree_util.tree_map(lambda x, y: jnp.sum(x*y), v, product)
    leaf_dots = tree_util.tree_leaves(leaf_dots_tree)
    return jnp.sum(jnp.array(leaf_dots)), v
  
  # NEW!!#TODO: JIT THIS!



  eta = step_size

  @dataclasses.dataclass
  class PlotData:
    # pylint: disable=g-bare-generic
    training_times: list
    eigenvalues: list
    sam_edges: list
    g_alignments: list
    sg_alignments: list
    training_losses: list
  plot_data = PlotData(list(),
                       list(),
                       list(),
                       list(),
                       list(),
                       list())
  if num_principal_comps > 1:
    for i in range(num_principal_comps):
      plot_data.eigenvalues.append(list())
  ce = hessian_norm.CurvatureEstimator(loss_by_params, rng)

  print("starting training", flush=True)
  start_time = time.time()
  last_hessian_check = start_time
  time_limit = 3600*time_limit_in_hours
  this_loss = None
  for x, y in train_batches:
    if ((time.time() > start_time + time_limit)
        or (this_loss and jnp.isnan(this_loss))):
      break

    if (hessian_check_gap and
        (time.time() > last_hessian_check + 3600.0*hessian_check_gap)):
      original_gradient = grad(loss_by_params)(params, x, y)
      sam_gradient = get_sam_gradient(params, x, y) if second_order == False else get_second_order_sam_gradient(params, x, y)
      if num_principal_comps == 1:
        curvature, principal_dir = ce.curvature_and_direction(params, x, y)
        this_hessian_norm = jnp.abs(curvature)
      else:
        print("calculating principal components", flush=True)
        eigs, principal_dir = ce.hessian_top_eigenvalues(params, x, y,
                                                         num_principal_comps)
        print("done calculating principal components", flush=True)
        this_hessian_norm = eigs[0]
      grad_hessian_alignment = mtu.get_alignment(original_gradient,
                                                 principal_dir)
      samgrad_hessian_alignment = mtu.get_alignment(sam_gradient,
                                                    principal_dir)
      training_time = time.time() - start_time
      original_gradient_norm = mtu.get_vector_norm(original_gradient)
      this_loss = loss_by_params(params, x, y)

      if rho == 0.0:
        sam_edge = 2.0/eta
      else:
        sam_edge = ((original_gradient_norm/(2.0*rho))
                    *(math.sqrt(1.0
                                + 8.0*rho/(eta*original_gradient_norm))
                      - 1.0))
      print("--------------", flush=True)
      formatting_string = ("Norm: {}, "
                           + "2/eta: {}, "
                           + "sam_edge: {}, "
                           + "|| g || = {}, "
                           + "loss = {}, "
                           + "g_alignment = {}, "
                           + "sg_alignment = {}")
      print(formatting_string.format(this_hessian_norm,
                                     2.0/eta,
                                     sam_edge,
                                     original_gradient_norm,
                                     this_loss,
                                     grad_hessian_alignment,
                                     samgrad_hessian_alignment, flush=True))
      if num_principal_comps > 1:
        print("eigs = {}".format(eigs, flush=True))
      if eigs_curve_filename or eigs_se_only_filename:
        plot_data.training_times.append(training_time)
        if num_principal_comps == 1:
          plot_data.eigenvalues.append(this_hessian_norm)
        else:
          for i in range(num_principal_comps):
            plot_data.eigenvalues[i].append(eigs[i])
        plot_data.sam_edges.append(sam_edge)
        plot_data.g_alignments.append(grad_hessian_alignment)
        plot_data.sg_alignments.append(samgrad_hessian_alignment)
        plot_data.training_losses.append(this_loss)
      if raw_data_filename:
        with open(raw_data_filename, "a") as raw_data_file:
          columns = [training_time,
                     this_hessian_norm,
                     2.0/eta,
                     sam_edge,
                     original_gradient_norm,
                     this_loss,
                     grad_hessian_alignment,
                     samgrad_hessian_alignment]
          format_string = "{} "*(len(columns)-1) + "{}\n"
          raw_data_file.write(format_string.format(*columns))
      last_hessian_check = time.time()

    params = update(params, x, y, eta)

  if (plot_data.sam_edges
      and (not jnp.isnan(jnp.array(plot_data.training_losses)).any())):
    if eigs_curve_filename:
      plt.figure()
      if num_principal_comps == 1:
        max_y = max(2.0/eta,
                    max(plot_data.eigenvalues),
                    max(plot_data.sam_edges))
      else:
        max_y = max(2.0/eta, max(plot_data.sam_edges))
        for i in range(num_principal_comps):
          max_y = max(max_y, max(plot_data.eigenvalues[i]))
      plt.ylim(0, 1.1*max_y)
      if num_principal_comps == 1:
        plt.plot(plot_data.training_times,
                 plot_data.eigenvalues,
                 color="b",
                 label="$|| H ||_{op}$")
      else:
        for i in range(num_principal_comps):
          plt.plot(plot_data.training_times,
                   plot_data.eigenvalues[i],
                   color="b",
                   label="$\lambda_{}$".format(i+1))
      plt.plot(plot_data.training_times,
               plot_data.sam_edges,
               color="g",
               label="SAM edge")
      plt.axhline(2.0/eta, color="m", label="$2/\eta$")
      plt.legend()
      plt.savefig(eigs_curve_filename, format="pdf", dpi=DPI)
    if eigs_se_only_filename:
      plt.figure()
      if num_principal_comps == 1:
        max_y = max(max(plot_data.eigenvalues), max(plot_data.sam_edges))
      else:
        max_y = max(plot_data.sam_edges)
        for i in range(num_principal_comps):
          max_y = max(max_y, max(plot_data.eigenvalues[i]))
      plt.ylim(0, 1.1*max_y)
      if num_principal_comps == 1:
        plt.plot(plot_data.training_times,
                 plot_data.eigenvalues,
                 color="b",
                 label="$|| H ||_{op}$")
      else:
        for i in range(num_principal_comps):
          plt.plot(plot_data.training_times,
                   plot_data.eigenvalues[i],
                   color="b",
                   label="$\lambda_{}$".format(i+1))
      plt.plot(plot_data.training_times,
               plot_data.sam_edges,
               color="g",
               label="SAM edge")
      plt.legend()

      plt.savefig(eigs_se_only_filename, format="pdf", dpi=DPI)
    if alignment_curve_filename:
      plt.figure()
      plt.ylim(0, 1.1*max(jnp.max(jnp.array(plot_data.g_alignments)),
                          jnp.max(jnp.array(plot_data.sg_alignments))))
      plt.plot(plot_data.training_times,
               plot_data.g_alignments,
               color="b",
               label="original gradient alignments")
      plt.plot(plot_data.training_times,
               plot_data.sg_alignments,
               color="g",
               label="SAM gradient alignments")
      plt.legend()

      plt.savefig(alignment_curve_filename, format="pdf", dpi=DPI)
    if loss_curve_filename:
      plt.figure()
      plt.ylim(0, 1.1*jnp.max(jnp.array(plot_data.training_losses)))
      plt.plot(plot_data.training_times,
               plot_data.training_losses,
               color="b",
               label="training loss")
      plt.legend()

      plt.savefig(loss_curve_filename, format="pdf", dpi=DPI)

  return params
