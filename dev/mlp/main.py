import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

# import matplotlib.pyplot as plt
import argparse

from model import *


def read_data(filename, num_lqr=0, num_ref=0):
	f = open(filename)
	data = json.load(f)
	f.close()
	states = np.array(data['states'], dtype="float32")
	inputs = np.array(data['inputs'], dtype="float32")
	next_states = np.array(data['nextstates'], dtype="float32")
	jacobians = np.swapaxes(np.array(data['jacobians'], dtype="float32"),1,2)
	num_lqr0 = data['num_lqr']
	num_ref0 = data['num_ref']
	samples = states.shape[0]
	N = samples // (num_lqr0 + num_ref0)
	print("N = {}".format(N))

	if num_lqr > 0 or num_ref > 0:
		states_lqr = states[:num_lqr0*N]
		inputs_lqr = inputs[:num_lqr0*N]
		jacobians_lqr = jacobians[:num_lqr0*N]
		states_ref = states[num_lqr0*N:]
		inputs_ref = inputs[num_lqr0*N:]
		jacobians_ref = jacobians[num_lqr0*N:]

		if num_lqr > 0:
			states_lqr = states_lqr[:num_lqr*N]
			inputs_lqr = inputs_lqr[:num_lqr*N]
			jacobians_lqr = jacobians_lqr[:num_lqr*N]
			num_lqr0 = num_lqr

		if num_ref > 0:
			states_ref = states_ref[:num_ref*N]
			inputs_ref = inputs_ref[:num_ref*N]
			jacobians_ref = jacobians_ref[:num_ref*N]
			num_ref0 = num_ref
		
		ipdb.set_trace()
		states = np.concatenate((states_lqr, states_ref))
		inputs = np.concatenate((inputs_lqr, inputs_ref))
		jacobians = np.concatenate((jacobians_lqr, jacobians_ref))

	return states, inputs, next_states, jacobians, num_lqr0, num_ref0


def train_model(filename='double_integrator.json', outfile="model_data.json", 
		num_epochs=100, alpha=0.5, num_lqr=0, num_ref=0,
		use_jacobian_regularization=False, hidden_dim=64, verbose=False):

	bsz = 512
	lr = 1e-3
	states, actions, next_states, jacobians, num_lqr, num_ref = read_data(filename, 
		num_lqr=num_lqr, num_ref=num_ref)
	num_state = states.shape[-1] 
	num_actions = actions.shape[-1]
	num_outs = num_state

	print("Training MLP")
	print("  Using data from \"{}\"".format(filename))
	print("  Use Jacobian regularization? {}".format(use_jacobian_regularization))
	print("  Alpha = {}".format(alpha))
	print("  Num epochs = {}".format(epochs))
	print("  Hidden dim = {}".format(hidden_dim))
	print("  Num lqr = {}".format(num_lqr))
	print("  Num ref = {}".format(num_ref))
	print("  Num states = {}\n  Num inputs = {}".format(num_state, num_actions))
	print("  Num samples = {}".format(states.shape[0]))
	print("  Saving to \"{}\"".format(outfile))

	states, actions, next_states, jacobians = torch.tensor(states), torch.tensor(actions), torch.tensor(next_states), torch.tensor(jacobians)
	out_space = Spaces((num_outs,), next_states.max(dim=0)[0], next_states.min(dim=0)[0])

	# Create Model
	model = DeterministicNetwork(num_state, num_actions, num_outs, hidden_dim, out_space)
		
	optim = torch.optim.Adam(model.parameters(), lr=lr)
	num_batches = states.shape[0]//bsz
	print("starting training")
	loss = 0.0
	for j in range(num_epochs):
		loss_average = 0.0
		for i in range(num_batches):
			state_batch = states[i*bsz:(i+1)*bsz]
			action_batch = actions[i*bsz:(i+1)*bsz]
			next_state_batch = next_states[i*bsz:(i+1)*bsz]
			jacobians_batch = jacobians[i*bsz:(i+1)*bsz]
			state_action = torch.cat([state_batch, action_batch], dim=-1).requires_grad_(True)
			pred_next_state = model(state_action)
			pred_jacobians = torch.stack([torch.autograd.grad(pred_next_state[:,i].sum(), state_action, retain_graph=True, create_graph=True)[0] for i in range(pred_next_state.shape[1])], dim=1)
			loss = (1 - alpha) * F.mse_loss(pred_next_state, next_state_batch)
			if use_jacobian_regularization:
				loss += alpha * F.mse_loss(pred_jacobians, jacobians_batch)
			optim.zero_grad()
			loss.backward()
			optim.step()
			loss_average += loss.item()

		loss = loss_average / num_batches
		if verbose:
			print(f"epoch : {j}, loss : {loss}")
		model.save_wts(outfile)
	print(f"Final Loss : {loss_average / num_batches}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="train an MLP")
	parser.add_argument('filename', 
		help="json file with the training data",
		type=str,
	)
	parser.add_argument('-o', '--output',
		help="Output file", type=str, 
		default="model_data.json",
	)
	parser.add_argument('--jacobian',
		action="store_true",
		help="Use Jacobian regularization",
		default=False
	)
	parser.add_argument('--verbose',
		action="store_true",
		help="Print out loss at each epoch",
		default=False
	)
	parser.add_argument('--alpha', type=float, default=0.5)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--hidden', type=int, default=64)
	parser.add_argument('--lqr', type=int, default=0)
	parser.add_argument('--ref', type=int, default=0)
	args = parser.parse_args()
	filename = args.filename
	outfile = args.output
	use_jacobian_regularization = args.jacobian
	alpha = args.alpha
	epochs = args.epochs
	hidden_dim = args.hidden 
	num_lqr = args.lqr
	num_ref = args.ref
	verbose = args.verbose

	train_model(filename, outfile, num_epochs=epochs, alpha=alpha, 
		use_jacobian_regularization=use_jacobian_regularization, 
		hidden_dim=hidden_dim,
		num_lqr=num_lqr, num_ref=num_ref,
		verbose=verbose
	)