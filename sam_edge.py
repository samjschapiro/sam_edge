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
import tensorflow as tf
import tensorflow_datasets as tfds

EPSILON = 1e-5
DPI = 300
n_iter_ = 25



def train(params,
          model,
          second_order,
          loss,
          epochs,
          steps_per_epoch,
          train_batches,
          test_batches,
          step_size,
          rho,  # for SAM -- if rho is 0.0, SAM is not used
          hessian_check_gap,
          raw_data_filename,
          num_principal_comps,
          rng):
  
  # def test_error_fn(params_, batches):
  #   @jit
  #   def batch_error(images, targets):
  #     predicted_class = jnp.argmax(model.apply(params_, images), axis=1)
  #     return jnp.mean(predicted_class != targets)

  #   sum_batch_errors = 0.0
  #   num_batches = 0
  #   for batch in batches:
  #     x, y = batch
  #     sum_batch_errors += batch_error(x, y)
  #     num_batches += 1
  #   return sum_batch_errors/num_batches
  
  @jit
  def loss_by_params(params, x_batched, y_batched):
    preds = model.apply(params, x_batched)
    return jnp.mean(loss(x_batched, preds, y_batched))
  
  def abs_loss(logits, y):
    return jnp.abs(logits - y)

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

################################################################
  eta = step_size
  ce = hessian_norm.CurvatureEstimator(loss_by_params, rng)

  print("starting training", flush=True)
  this_loss = None
  
  epoch = 0
  for x, y in train_batches:
    if epoch % hessian_check_gap == 0:
      this_loss = loss_by_params(params, x, y)
      # test_err = test_error_fn(params, test_batches)
      original_gradient = grad(loss_by_params)(params, x, y)
      sam_gradient = get_sam_gradient(params, x, y)
      if epoch == 0:
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

      if second_order:
        ssam_gradient = get_ssam_gradient(params, x, y, n_iter=n_iter_, beta_start=sam_gradient)
        ssamgrad_hessian_alignment = mtu.get_alignment(ssam_gradient,
                                              principal_dir)
        ssam_sam_grads_alignment = mtu.get_alignment(sam_gradient, ssam_gradient)   

      grad_hessian_alignment = mtu.get_alignment(original_gradient,
                                                 principal_dir)
      samgrad_hessian_alignment = mtu.get_alignment(sam_gradient,
                                                    principal_dir)
      sam_succ_grad_alignment = mtu.get_alignment(sam_gradient, prev_sam_gradient)
      succ_grad_alignment = mtu.get_alignment(original_gradient, prev_original_gradient)
      prev_sam_gradient = sam_gradient
      prev_original_gradient = original_gradient
      print("--------------", flush=True)
      if second_order:
        formatting_string = ("Epoch = {}, "
                      + "Train Loss = {}, "
                      # + "Test Loss = {}, "
                      + "lambda1: {}, "
                      + "2/eta: {}, "
                      + "g_alignment = {}, "
                      + "sg_alignment = {}, "
                      + "ssg_alignment = {}, "
                      + "ssam_sam_g_alignment = {}")
        print(formatting_string.format(epoch,
                                      this_loss,
                                      # test_err,
                                      this_hessian_norm,
                                      2.0/eta,
                                      grad_hessian_alignment,
                                      samgrad_hessian_alignment, 
                                      ssamgrad_hessian_alignment,
                                      ssam_sam_grads_alignment, 
                                      flush=True))
      else:
        formatting_string = ("Epoch = {}, "
                            + "Train Loss = {}, "
                            + "Test Loss = {}, "
                            + "lambda1: {}, "
                            + "2/eta: {}, "
                            + "g_alignment = {}, "
                            + "sg_alignment = {},"
                            + "succ_g_alignment = {},"
                            + "succ_sg_alignment = {}")
        print(formatting_string.format(epoch,
                                      this_loss,
                                      # test_err,
                                      this_hessian_norm,
                                      2.0/eta,
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
            columns = [epoch,
                      this_loss,
                      # test_err,
                      this_hessian_norm,
                      2.0/eta,
                      grad_hessian_alignment,
                      samgrad_hessian_alignment,
                      ssamgrad_hessian_alignment,
                      ssam_sam_grads_alignment]
            format_string = "{} "*(len(columns)-1) + "{}\n"
            raw_data_file.write(format_string.format(*columns))
          else:
            columns = [epoch,
                      this_loss,
                      # test_err, 
                      this_hessian_norm,
                        2.0/eta,
                        grad_hessian_alignment,
                        samgrad_hessian_alignment,
                        succ_grad_alignment,
                        sam_succ_grad_alignment]
            format_string = "{} "*(len(columns)-1) + "{}\n"
            raw_data_file.write(format_string.format(*columns))

    if second_order:
      params = ssam_update(params, x, y, eta, n_iter=n_iter_)
    else:
      params = update(params, x, y, eta)
    epoch += 1/steps_per_epoch
    print(epoch)
  return params
