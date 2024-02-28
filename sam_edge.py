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
second_order = True

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

  @jit
  def loss_by_params(params, x_batched, y_batched):
    preds = model.apply(params, x_batched)
    return jnp.mean(loss(x_batched, preds, y_batched))
  
  def abs_loss(logits, y):
    return jnp.abs(logits - y)
  
  @jit
  def apply_model(params, x_batched):
    return 0.1+jnp.sum(jnp.argmax(model.apply(params, x_batched), axis=1))-0.1


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
  def update(params, x, y, eta):
    if rho > 0.0:
      grad_location = sam_neighbor(params, x, y) 
    else:
      grad_location = params
    grads = grad(loss_by_params)(grad_location, x, y)
    return tree_util.tree_map(lambda p, g: p - eta * g,
                              params,
                              grads)
  
  def second_sam_update(params, x, y, n_iter, beta_start):
    if rho > 0.0:
        betas = beta_start
        # SSAM objective function
        def ssam_func(beta, params_, x_, y_):
           grads = grad(loss_by_params)(params_, x_, y_)
           func_grads = grad(apply_model2)(params_, x_, y_)
           return jnp.sum(jnp.array(tree_util.tree_leaves(tree_util.tree_map(lambda b, g, f: jnp.sum(-b*g - (b*f)**2), # MSE loss, so l_i'' is 1
                                    beta,
                                    grads,
                                    func_grads))))
        pga = jo.ProjectedGradient(fun=ssam_func, projection=jo.projection.projection_l2_ball, stepsize=rho, maxiter=n_iter)
        beta_star, _ = pga.run(betas, hyperparams_proj=rho, params_=params, x_=x, y_=y)
        grad_location = tree_util.tree_map(lambda p, b: p + b, 
                                  params, 
                                  beta_star)
        grads = grad(loss_by_params)(grad_location, x, y)
        return tree_util.tree_map(lambda p, g: p - eta * g,
                          params,
                          grads), grads
    else:
      grad_location = params
      grads = grad(loss_by_params)(grad_location, x, y)
      return tree_util.tree_map(lambda p, g: p - eta * g,
                                params,
                                grads), grads

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
      sam_gradient = get_sam_gradient(params, x, y)
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
        _, ssam_gradient = second_sam_update(params, x, y, n_iter=15, beta_start=sam_gradient)
        ssamgrad_hessian_alignment = mtu.get_alignment(ssam_gradient,
                                              principal_dir)
        ssam_gradient_norm = mtu.get_vector_norm(ssam_gradient)
        ssam_gradient_l1_norm = mtu.get_vector_l1_norm(ssam_gradient)

      grad_hessian_alignment = mtu.get_alignment(original_gradient,
                                                 principal_dir)
      samgrad_hessian_alignment = mtu.get_alignment(sam_gradient,
                                                    principal_dir)
     

      training_time = time.time() - start_time
      original_gradient_norm = mtu.get_vector_norm(original_gradient)
      sam_gradient_norm = mtu.get_vector_norm(sam_gradient)
      original_gradient_l1_norm = mtu.get_vector_l1_norm(original_gradient)
      sam_gradient_l1_norm = mtu.get_vector_l1_norm(sam_gradient)
      this_loss = loss_by_params(params, x, y)

      if rho == 0.0:
        sam_edge = 2.0/eta
      else:
        sam_edge = ((original_gradient_norm/(2.0*rho))
                    *(math.sqrt(1.0
                                + 8.0*rho/(eta*original_gradient_norm))
                      - 1.0))
      print("--------------", flush=True)
      if second_order:
        formatting_string = ("Loss = {}, "
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
                      + "ssg_alignment = {}")
        print(formatting_string.format(this_loss,
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
                                      flush=True))
      else:
        formatting_string = ("Loss = {}, "
                            + "lambda1: {}, "
                            + "2/eta: {}, "
                            + "sam_edge: {}, "
                            + "|| g ||_2 = {}, "
                            + "|| g_sam ||_2 = {}, "
                            + "|| g ||_1 = {}, "
                            + "|| g_sam||_1 = {}, "
                            + "g_alignment = {}, "
                            + "sg_alignment = {}")
        print(formatting_string.format(this_loss,
                                      this_hessian_norm,
                                      2.0/eta,
                                      sam_edge,
                                      original_gradient_norm,
                                      sam_gradient_norm,
                                      original_gradient_l1_norm,
                                      sam_gradient_l1_norm,
                                      grad_hessian_alignment,
                                      samgrad_hessian_alignment, flush=True))
      if num_principal_comps > 1:
        print("eigs = {}".format(eigs, flush=True))
      # if eigs_curve_filename or eigs_se_only_filename:
      #   plot_data.training_times.append(training_time)
      #   if num_principal_comps == 1:
      #     plot_data.eigenvalues.append(this_hessian_norm)
      #   else:
      #     for i in range(num_principal_comps):
      #       plot_data.eigenvalues[i].append(eigs[i])
      #   plot_data.sam_edges.append(sam_edge)
      #   plot_data.g_alignments.append(grad_hessian_alignment)
      #   plot_data.sgd_gradients.append(original_gradient_norm)
      #   plot_data.sam_gradients.append(sam_gradient_norm)
      #   plot_data.sam_grad_unif_kl.append(sam_gradient_l1_norm)
      #   plot_data.sgd_gradient_unif_kl.append(original_gradient_l1_norm)
      #   plot_data.sg_alignments.append(samgrad_hessian_alignment)
      #   plot_data.training_losses.append(this_loss)
      if raw_data_filename:
        with open(raw_data_filename, "a") as raw_data_file:
          if second_order: 
            columns = [this_loss,
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
                      ssamgrad_hessian_alignment]
            format_string = "{} "*(len(columns)-1) + "{}\n"
            raw_data_file.write(format_string.format(*columns))
          else:
            columns = [this_loss,
                      this_hessian_norm,
                        2.0/eta,
                        sam_edge,
                        original_gradient_norm,
                        sam_gradient_norm,
                        original_gradient_l1_norm,
                        sam_gradient_l1_norm,
                        grad_hessian_alignment,
                        samgrad_hessian_alignment]
            format_string = "{} "*(len(columns)-1) + "{}\n"
            raw_data_file.write(format_string.format(*columns))
      last_hessian_check = time.time()

  if not second_order:  
    params = update(params, x, y, eta)

      

  # if (plot_data.sam_edges
  #     and (not jnp.isnan(jnp.array(plot_data.training_losses)).any())):
  #   if eigs_curve_filename:
  #     plt.figure()
  #     if num_principal_comps == 1:
  #       max_y = max(2.0/eta,
  #                   max(plot_data.eigenvalues),
  #                   max(plot_data.sam_edges))
  #     else:
  #       max_y = max(2.0/eta, max(plot_data.sam_edges))
  #       max_y = 0
  #       for i in range(num_principal_comps):
  #         max_y = max(max_y, max(plot_data.eigenvalues[i]))
  #     plt.ylim(0, 1.1*max_y)
  #     if num_principal_comps == 1:
  #       plt.plot(plot_data.training_times,
  #                plot_data.eigenvalues,
  #                color="b",
  #                label="$|| H ||_{op}$")
  #     else:
  #       for i in range(num_principal_comps):
  #         # colors: b, g, r, c, m, y, k
  #         colormap = {0: 'b', 1: 'g', 2: 'r', 3: 'c', 4: 'm', 5:'y', 6:'k'}
  #         clr = colormap[i%7]
  #         plt.plot(plot_data.training_times,
  #                  plot_data.eigenvalues[i],
  #                  color=clr,
  #                  label="$\lambda_{}$".format(i+1))
  #     plt.plot(plot_data.training_times,
  #              plot_data.sam_edges,
  #              color="g",
  #              label="SAM edge")
  #     plt.axhline(2.0/eta, color="m", label="$2/\eta$")
  #     plt.legend()
  #     plt.savefig(eigs_curve_filename, format="pdf", dpi=DPI)
  #   if eigs_se_only_filename:
  #     plt.figure()
  #     if num_principal_comps == 1:
  #       max_y = max(max(plot_data.eigenvalues), max(plot_data.sam_edges))
  #     else:
  #       max_y = max(plot_data.sam_edges)
  #       for i in range(num_principal_comps):
  #         max_y = max(max_y, max(plot_data.eigenvalues[i]))
  #     plt.ylim(0, 1.1*max_y)
  #     if num_principal_comps == 1:
  #       plt.plot(plot_data.training_times,
  #                plot_data.eigenvalues,
  #                color="b",
  #                label="$|| H ||_{op}$")
  #     else:
  #       for i in range(num_principal_comps):
  #         plt.plot(plot_data.training_times,
  #                  plot_data.eigenvalues[i],
  #                  color="b",
  #                  label="$\lambda_{}$".format(i+1))
  #     plt.plot(plot_data.training_times,
  #              plot_data.sam_edges,
  #              color="g",
  #              label="SAM edge")
  #     plt.legend()

  #     plt.savefig(eigs_se_only_filename, format="pdf", dpi=DPI)
  #   if alignment_curve_filename:
  #     plt.figure()
  #     plt.ylim(0, 1.1*max(jnp.max(jnp.array(plot_data.g_alignments)),
  #                         jnp.max(jnp.array(plot_data.sg_alignments))))
  #     plt.plot(plot_data.training_times,
  #              plot_data.g_alignments,
  #              color="b",
  #              label="original gradient alignments")
  #     plt.plot(plot_data.training_times,
  #              plot_data.sg_alignments,
  #              color="g",
  #              label="SAM gradient alignments")
  #     plt.legend()

  #     plt.savefig(alignment_curve_filename, format="pdf", dpi=DPI)
  #   if loss_curve_filename:
  #     plt.figure()
  #     plt.ylim(0, 1.1*jnp.max(jnp.array(plot_data.training_losses)))
  #     plt.plot(plot_data.training_times,
  #              plot_data.training_losses,
  #              color="b",
  #              label="training loss")
  #     plt.legend()

  #     plt.savefig(loss_curve_filename, format="pdf", dpi=DPI)
  #   if sam_grad_norm_output:
  #     plt.figure()
  #     sam_grads = jnp.max(jnp.array(plot_data.sam_gradients))
  #     sgd_grads =  jnp.max(jnp.array(plot_data.sgd_gradients))
  #     ylim = sam_grads if sam_grads >= sgd_grads else sgd_grads
  #     plt.ylim(0, 1.1*ylim)
  #     plt.plot(plot_data.training_times,
  #              plot_data.sam_gradients,
  #              color="b",
  #              label="$\| g_{sam} \|$")
  #     plt.plot(plot_data.training_times,
  #             plot_data.sgd_gradients,
  #             color="r",
  #             label="$\| g \|$")
  #     plt.legend()

  #     plt.savefig(sam_grad_norm_output, format="pdf", dpi=DPI)

  #   if grad_unif_kl_output:
  #     plt.figure()
  #     sam_kl = jnp.max(jnp.array(plot_data.sam_grad_unif_kl))
  #     sgd_kl = jnp.max(jnp.array(plot_data.sgd_gradient_unif_kl))
  #     ylim = sam_kl if sam_kl >= sgd_kl else sgd_kl
  #     plt.ylim(0, 1.1*ylim)
  #     plt.plot(plot_data.training_times,
  #              plot_data.sam_grad_unif_kl,
  #              color="b",
  #              label="$\|g_{sam}\|_1$")
  #     plt.plot(plot_data.training_times,
  #             plot_data.sgd_gradient_unif_kl,
  #             color="r",
  #             label="$\|g\|_1$")
  #     plt.legend()

  #     plt.savefig(grad_unif_kl_output, format="pdf", dpi=DPI)

  return params
