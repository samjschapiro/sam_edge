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
from jax import vmap
import jax.random as jrandom
import jaxopt as jo
import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import more_tree_utils as mtu

EPSILON = 1e-5
DPI = 300

# pylint: disable=anomalous-backslash-in-string


def train(params,
          model,
          second_order,
          loss,
          epochs,
          train_batches,
          step_size,
          rho,  # for SAM -- if rho is 0.0, SAM is not used
          hessian_check_gap,
          eigs_curve_filename,
          eigs_se_only_filename,
          alignment_curve_filename,
          loss_curve_filename,
          raw_data_filename,
          sam_grad_norm_output,
          grad_unif_kl_output,
          num_principal_comps,
          time_limit_in_hours,
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
  second_order = False
  @jit
  def loss_by_params(params, x_batched, y_batched):
    preds = model.apply(params, x_batched)
    return jnp.mean(loss(x_batched, preds, y_batched))
  
  def abs_loss(logits, y):
    return jnp.abs(logits - y)
  
  # @jit
  # def apply_model(params, x_batched):
  #   return 0.1+jnp.sum(jnp.argmax(model.apply(params, x_batched), axis=1))-0.1


  def apply_model2(params, x_batched, y_batched):
    preds = model.apply(params, x_batched)
    return jnp.mean(abs_loss(preds, y_batched))
  
  @jit
  def sam_neighbor(params, x, y):
    grads = grad(loss_by_params)(params, x, y)
    norm = mtu.get_vector_norm(grads)
    return tree_util.tree_map(lambda p, g: p + rho * g/(norm + EPSILON),
                              params,
                              grads)
  
  @jit
  def get_ssam_gradient(params, x, y, n_iter, beta_start):
    betas = beta_start
    # SSAM objective function
    def ssam_func(beta, params_, x_, y_):
        grads = grad(loss_by_params)(params_, x_, y_)
        func_grads = grad(apply_model2)(params_, x_, y_)
        return jnp.sum(jnp.array(tree_util.tree_leaves(tree_util.tree_map(lambda b, nabla_l, nabla_f: jnp.sum(-b*nabla_l - (b*nabla_f)**2), # MSE loss, so l_i'' is 1 #
                                beta,
                                grads,
                                func_grads))))
    pga = jo.ProjectedGradient(fun=ssam_func, projection=jo.projection.projection_l2_ball, stepsize=rho/2, maxiter=n_iter)
    beta_star, _ = pga.run(betas, hyperparams_proj=rho, params_=params, x_=x, y_=y)
    grad_location = tree_util.tree_map(lambda p, b: p + b, 
                              params, 
                              beta_star)
    grads = grad(loss_by_params)(grad_location, x, y)
    return grads 

  @jit
  def ssam_neighbor(params, x, y, n_iter, beta_start):
    grads = get_ssam_gradient(params, x, y, n_iter, beta_start)
    return tree_util.tree_map(lambda p, g: p - eta * g,
                      params,
                      grads)


  @jit
  def update(params, x, y, eta):
    if rho > 0.0:
      grad_location = sam_neighbor(params, x, y) 
    else:
      grad_location = params
    grads = grad(loss_by_params)(grad_location, x, y)
    return tree_util.tree_map(lambda p, g: p - eta * g,
                              params,
                              grads)
  @jit
  def ssam_update(params, x, y, eta, n_iter=5):
    if rho > 0.0:
      beta_start = sam_neighbor(params, x, y)
      grad_location = ssam_neighbor(params, x, y, n_iter, beta_start) # Run projected gradient ascent to get SSAM neightbor
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

  eta = step_size

  @dataclasses.dataclass
  class PlotData:
    # pylint: disable=g-bare-generic
    training_times: list
    eigenvalues: list
    sam_edges: list
    g_alignments: list
    sam_gradients: list
    sam_grad_unif_kl: list
    sgd_gradients: list
    sgd_gradient_unif_kl: list
    sg_alignments: list
    training_losses: list
  plot_data = PlotData(list(),
                       list(),
                       list(),
                       list(),
                       list(),
                       list(),
                       list(),
                       list(),
                       list(),
                       list())
  # if num_principal_comps > 1:
  #   for i in range(num_principal_comps):
  #     plot_data.eigenvalues.append(list())
  ce = hessian_norm.CurvatureEstimator(loss_by_params, rng)

  print("starting training", flush=True)
  this_loss = None
  it_num = 0
  for x, y in train_batches:
    if it_num == epochs:
      break
    if it_num % hessian_check_gap == 0:
      original_gradient = grad(loss_by_params)(params, x, y)
      sam_gradient = get_sam_gradient(params, x, y)
      if it_num == 0:
        prev_original_gradient = original_gradient
        prev_sam_gradient = sam_gradient
      if num_principal_comps == 1:
        curvature, principal_dir = ce.curvature_and_direction(params, x, y)
        this_hessian_norm = jnp.abs(curvature)
      else:
        print("calculating principal components", flush=True)
        eigs, principal_dir = ce.hessian_top_eigenvalues(params, x, y,
                                                         num_principal_comps)
        print("done calculating principal components", flush=True)
        this_hessian_norm = eigs[0]
      this_loss = loss_by_params(params, x, y)
      if second_order:
        ssam_gradient = get_ssam_gradient(params, x, y, n_iter=5, beta_start=sam_gradient)
        ssamgrad_hessian_alignment = mtu.get_alignment(ssam_gradient,
                                              principal_dir)
        ssam_gradient_norm = mtu.get_vector_norm(ssam_gradient)
        ssam_gradient_l1_norm = mtu.get_vector_l1_norm(ssam_gradient)
        ssam_sam_grads_alignment = mtu.get_alignment(sam_gradient, ssam_gradient)   

      grad_hessian_alignment = mtu.get_alignment(original_gradient,
                                                 principal_dir)
      samgrad_hessian_alignment = mtu.get_alignment(sam_gradient,
                                                    principal_dir)
      sam_succ_grad_alignment = mtu.get_alignment(sam_gradient, prev_sam_gradient)
      succ_grad_alignment = mtu.get_alignment(original_gradient, prev_original_gradient)
      prev_sam_gradient = sam_gradient
      prev_original_gradient = original_gradient
      original_gradient_norm = mtu.get_vector_norm(original_gradient)
      sam_gradient_norm = mtu.get_vector_norm(sam_gradient)
      original_gradient_l1_norm = mtu.get_vector_l1_norm(original_gradient)
      sam_gradient_l1_norm = mtu.get_vector_l1_norm(sam_gradient)

      if rho == 0.0:
        sam_edge = 2.0/eta
      else:
        sam_edge = ((original_gradient_norm/(2.0*rho))
                    *(math.sqrt(1.0
                                + 8.0*rho/(eta*original_gradient_norm))
                      - 1.0))
      print("--------------", flush=True)
      if second_order:
        formatting_string = ("Epoch = {}, "
                      + "Loss = {}, "
                      + "lambda1: {}, "
                      + "2/eta: {}, "
                      + "sam_edge: {}, "
                      + "|| g ||_2 = {}, "
                      + "|| g_sam ||_2 = {}, "
                      + "|| g_ssam ||_2 = {}, "
                      + "|| g ||_1 = {}, "
                      + "|| g_sam||_1 = {}, "
                      + "|| g_ssam||_1 = {}, "
                      + "g_alignment = {}, "
                      + "sg_alignment = {}, "
                      + "ssg_alignment = {}, "
                      + "ssam_sam_g_alignment = {}")
        print(formatting_string.format(it_num,
                                      this_loss,
                                      this_hessian_norm,
                                      2.0/eta,
                                      sam_edge,
                                      original_gradient_norm,
                                      sam_gradient_norm,
                                      ssam_gradient_norm,
                                      original_gradient_l1_norm,
                                      sam_gradient_l1_norm,
                                      ssam_gradient_l1_norm,
                                      grad_hessian_alignment,
                                      samgrad_hessian_alignment, 
                                      ssamgrad_hessian_alignment,
                                      ssam_sam_grads_alignment, 
                                      flush=True))
      else:
        formatting_string = ("Epoch = {}, "
                            + "Loss = {}, "
                            + "lambda1: {}, "
                            + "2/eta: {}, "
                            + "sam_edge: {}, "
                            + "|| g ||_2 = {}, "
                            + "|| g_sam ||_2 = {}, "
                            + "|| g ||_1 = {}, "
                            + "|| g_sam||_1 = {}, "
                            + "g_alignment = {}, "
                            + "sg_alignment = {},"
                            + "succ_g_alignment = {},"
                            + "succ_sg_alignment = {}")
        print(formatting_string.format(it_num,
                                      this_loss,
                                      this_hessian_norm,
                                      2.0/eta,
                                      sam_edge,
                                      original_gradient_norm,
                                      sam_gradient_norm,
                                      original_gradient_l1_norm,
                                      sam_gradient_l1_norm,
                                      grad_hessian_alignment,
                                      samgrad_hessian_alignment, 
                                      succ_grad_alignment,
                                      sam_succ_grad_alignment,
                                      flush=True))
      if num_principal_comps > 1:
        print("eigs = {}".format(eigs, flush=True))
      if raw_data_filename:
        with open(raw_data_filename, "a") as raw_data_file:
          if second_order: 
            columns = [it_num,
                      this_loss,
                      this_hessian_norm,
                      2.0/eta,
                      sam_edge,
                      original_gradient_norm,
                      sam_gradient_norm,
                      ssam_gradient_norm,
                      original_gradient_l1_norm,
                      sam_gradient_l1_norm,
                      ssam_gradient_l1_norm,
                      grad_hessian_alignment,
                      samgrad_hessian_alignment,
                      ssamgrad_hessian_alignment,
                      ssam_sam_grads_alignment]
            format_string = "{} "*(len(columns)-1) + "{}\n"
            raw_data_file.write(format_string.format(*columns))
          else:
            columns = [it_num,
                      this_loss,
                      this_hessian_norm,
                        2.0/eta,
                        sam_edge,
                        original_gradient_norm,
                        sam_gradient_norm,
                        original_gradient_l1_norm,
                        sam_gradient_l1_norm,
                        grad_hessian_alignment,
                        samgrad_hessian_alignment,
                        succ_grad_alignment,
                        sam_succ_grad_alignment]
            format_string = "{} "*(len(columns)-1) + "{}\n"
            raw_data_file.write(format_string.format(*columns))

    it_num += 1
    n_iter_ = 15
    if second_order:
      params = ssam_update(params, x, y, eta, n_iter=n_iter_)
    else:
      params = update(params, x, y, eta)

    

  return params
