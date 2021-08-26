import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import model
import config
import evaluate
import evaluate_wholeItemsRank
import data_utils

from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--lr",
	type=float,
	default=0.001,
	help="learning rate")
parser.add_argument("--dropout",
	type=float,
	default=0.5,
	help="dropout rate")
parser.add_argument("--batch_size",
	type=int,
	default=256,
	help="batch size for training")
parser.add_argument("--run_name",
	type=str,
	default="log.txt",
	help="name of log of this run")
parser.add_argument("--epochs",
	type=int,
	default=200,
	help="training epoches")
parser.add_argument("--top_k",
	type=int,
	default=20,
	help="compute metrics@top_k")
parser.add_argument("--factor_num",
	type=int,
	default=64,
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers",
	type=int,
	default=3,
	help="number of layers in MLP model")
parser.add_argument("--num_ng",
	type=int,
	default=4,
	help="sample negative items for training")
parser.add_argument("--test_num_ng",
	type=int,
	default=99,
	help="sample part of negative items for testing")
parser.add_argument("--out",
	default=True,
	help="save model or not")
parser.add_argument("--gpu",
	type=str,
	default="0",
	help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


############################## PREPARE DATASET ##########################
#train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()
train_data, click_data, fav_data, cart_data, user_num ,item_num, train_mat, click_mat, fav_mat, cart_mat = data_utils.load_all()


# construct the train and test datasets
train_dataset = data_utils.NCFData(
		train_data, click_data, fav_data, cart_data, item_num, train_mat, click_mat, fav_mat, cart_mat, args.num_ng, True)
# test_dataset = data_utils.NCFData(
# 		test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)
# test_loader = data.DataLoader(test_dataset,
# 		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)


########################### CREATE MODEL #################################
if config.model == 'NeuMF-pre':
	assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
	assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(config.GMF_model_path)
	MLP_model = torch.load(config.MLP_model_path)
else:
	GMF_model = None
	MLP_model = None

model = model.NCF(user_num, item_num, args.factor_num, args.num_layers,
						args.dropout, config.model, GMF_model, MLP_model)
model.cuda()

weight_buy = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
weight_click = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
weight_fav = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
weight_cart = torch.tensor(torch.FloatTensor([1]), requires_grad=True)

weights = [weight_buy, weight_click, weight_fav, weight_cart]
opt1 = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0000001)
opt2 = torch.optim.Adam(weights, lr=args.lr)
loss_function = nn.BCEWithLogitsLoss()
Gradloss = nn.L1Loss()
alpha = 0.1#0.25, 0.75, 1.5
l0 = torch.log(torch.FloatTensor([2])).cuda() #When Li(0) is sharply dependent on initialization, we can use a theoretical initial loss instead. E.g. for Li
#the CE loss across C classes, we can use Li(0) = log(C).
# writer = SummaryWriter() # for visualization

########################### TRAINING #####################################
f = open(os.path.join(config.log_path, args.run_name), 'w')
count, best_hr = 0, 0

for epoch in range(args.epochs):
	model.train() # Enable dropout (if have).
	start_time = time.time()
	train_loader.dataset.ng_sample()

	#for user, item, label in train_loader:
	print('Training at this epoch start.')
	loss_epoch_buy = 0.0
	loss_epoch_click = 0.0
	loss_epoch_fav = 0.0
	loss_epoch_cart = 0.0
	for step, batch in enumerate(tqdm(train_loader)):
		user = batch[0]
		item = batch[1]
		label = batch[2]
		label_click = batch[3].float().cuda()
		label_fav = batch[4].float().cuda()
		label_cart = batch[5].float().cuda()
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()

		weights[0] = weights[0].cuda()
		weights[1] = weights[1].cuda()
		weights[2] = weights[2].cuda()	
		weights[3] = weights[3].cuda()		

		prediction_buy, prediction_click, prediction_fav, prediction_cart = model(user, item)
		loss_buy_origin = loss_function(prediction_buy, label)
		loss_click_origin = loss_function(prediction_click, label_click)
		loss_fav_origin = loss_function(prediction_fav, label_fav)
		loss_cart_origin = loss_function(prediction_cart, label_cart)

		loss_buy = loss_buy_origin*weights[0]
		loss_click = loss_click_origin*weights[1]
		loss_fav = loss_fav_origin*weights[2]
		loss_cart = loss_cart_origin*weights[3]
		loss = loss_buy + loss_click + loss_fav + loss_cart

		opt1.zero_grad()
		loss.backward(retain_graph=True)   

		# Getting gradients of the first layers of each tower and calculate their l2-norm 
		last_layer_shared = list(model.MLP_layers.parameters())[0]
		loss_buy_grad = torch.autograd.grad(loss_buy, last_layer_shared, retain_graph=True, create_graph=True)
		loss_buy_grad_norm = torch.norm(loss_buy_grad[0], 2)
		loss_click_grad = torch.autograd.grad(loss_click, last_layer_shared, retain_graph=True, create_graph=True)
		loss_click_grad_norm = torch.norm(loss_click_grad[0], 2)
		loss_fav_grad = torch.autograd.grad(loss_fav, last_layer_shared, retain_graph=True, create_graph=True)
		loss_fav_grad_norm = torch.norm(loss_fav_grad[0], 2)
		loss_cart_grad = torch.autograd.grad(loss_cart, last_layer_shared, retain_graph=True, create_graph=True)
		loss_cart_grad_norm = torch.norm(loss_cart_grad[0], 2)
		G_avg = torch.div((loss_buy_grad_norm + loss_click_grad_norm + loss_fav_grad_norm + loss_cart_grad_norm), 4)

		# Calculating relative losses 
		inv_rate_buy = torch.div(loss_buy,l0)
		inv_rate_click = torch.div(loss_click,l0)
		inv_rate_fav = torch.div(loss_fav,l0)
		inv_rate_cart = torch.div(loss_cart,l0)
		inv_rate_avg = torch.div((inv_rate_buy + inv_rate_click + inv_rate_fav + inv_rate_cart), 4)

		# Calculating relative inverse training rates for tasks
		rel_inv_rate_buy= torch.div(inv_rate_buy,inv_rate_avg)
		rel_inv_rate_click= torch.div(inv_rate_click,inv_rate_avg)
		rel_inv_rate_fav= torch.div(inv_rate_fav,inv_rate_avg)
		rel_inv_rate_cart= torch.div(inv_rate_cart,inv_rate_avg)

		# Calculating the constant target for Eq. 2 in the GradNorm paper
		C_buy = G_avg*(rel_inv_rate_buy)**alpha
		C_click = G_avg*(rel_inv_rate_click)**alpha
		C_fav = G_avg*(rel_inv_rate_fav)**alpha
		C_cart = G_avg*(rel_inv_rate_cart)**alpha
		C_buy = C_buy.detach()
		C_click = C_click.detach()
		C_fav = C_fav.detach()
		C_cart = C_cart.detach()

		opt2.zero_grad()
		# Calculating the gradient loss according to Eq. 2 in the GradNorm paper
		Lgrad = Gradloss(loss_buy_grad_norm, C_buy) + Gradloss(loss_click_grad_norm, C_click) + Gradloss(loss_fav_grad_norm, C_fav) + Gradloss(loss_cart_grad_norm, C_cart)
		Lgrad.backward()
		
		# Updating loss weights 
		opt2.step()

		# Updating the model weights # I think here is problem, because the grad of Lgrad will be accumulated in the last shared layer when we do Lgrad.backward()
		opt1.step()

		# Renormalizing the losses weights
		coef = 4/(weight_buy + weight_click + weight_fav + weight_cart)
		weights = [coef*weight_buy, coef*weight_click, coef*weight_fav, coef*weight_cart]

		loss_epoch_buy += loss_buy_origin
		loss_epoch_click += loss_click_origin
		loss_epoch_fav += loss_fav_origin
		loss_epoch_cart += loss_cart_origin

		count += 1
	print('buy loss of epoch: '+str(epoch), loss_epoch_buy*1.0/step)
	print('click loss of epoch: '+str(epoch), loss_epoch_click*1.0/step)
	print('fav loss of epoch: '+str(epoch), loss_epoch_fav*1.0/step)
	print('cart loss of epoch: '+str(epoch), loss_epoch_cart*1.0/step)
	print(weight_buy, weight_click, weight_fav, weight_cart)

	model.eval()

	t_validation = evaluate_wholeItemsRank.evaluateModel(model, user_num, item_num, args.top_k, epoch, 16, config.train_rating, config.validation_rating)
	f.write('Validation in epoch: '+ str(epoch) + ' loss: ' + str(loss_epoch_buy*1.0/step) + '\n' + str(t_validation) + '\n')#str(t_valid) + ' ' +
	f.flush()
	
	t_test = evaluate_wholeItemsRank.evaluateModel(model, user_num, item_num, args.top_k, epoch, 16, config.train_rating, config.test_rating)	
	f.write('Test in epoch: '+str(epoch)+'\n'+'loss: '+str(loss_epoch_buy*1.0/step)+'\n'+'click loss of epoch: '+str(loss_epoch_click*1.0/step)+'\n'+'fav loss of epoch: '+str(loss_epoch_fav*1.0/step)+'\n'+'cart loss of epoch: '+str(loss_epoch_cart*1.0/step)+'\n'+str(t_test) + '\n')
	f.flush()
	
	elapsed_time = time.time() - start_time
	print("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
		