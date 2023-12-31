from torch import nn
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import gc 


def v_wrap(np_array, dtype=np.float32):
    torch.set_num_threads(1)
    if isinstance(np_array, torch.Tensor):
        np_array = np_array.cpu()
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def push_and_pull(actoropt,criticopt, lnet, gnet, done, boardstate_, vectorstate_, buffer_boardstate, buffer_vectorstate, ba, br, gamma, device, global_device, total_step, won, logits, values):
    torch.set_num_threads(1)
    #if done:
    #    v_s_ = 0.               # terminal
    #else:
    #    v_s_ = lnet.forward(boardstate_, vectorstate_)[-1].data.cpu().numpy()[0, 0]
    v_s_ = 0.
    buffer_v_target = []

    if len(br) == 1:
        buffer_v_target.append(br[0] + gamma * v_s_)
    else:
        for r in br[::-1]:    # reverse buffer r
            v_s_ = r + gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

    

    buffer_boardstate_cpu = [tensor.cpu() for tensor in buffer_boardstate]
    buffer_vectorstate_cpu = [tensor.cpu() for tensor in buffer_vectorstate]

    total_a_loss = 0
    total_c_loss = 0
    total_entropy = 0
    total_l2_activity_loss = 0
    
    totalactorloss = 0
    totalcriticloss = 0

    actorloss = 0
    criticloss = 0

    #print("len(buffer_boardstate_cpu): ", len(buffer_boardstate_cpu))
    #print("len(buffer_vectorstate_cpu): ", len(buffer_vectorstate_cpu))
    #print("len(buffer_v_target): ", len(buffer_v_target))
    #print("len(ba): ", len(ba))
    #print("len(logits): ", len(logits))
    #print("len(values): ", len(values))
    

    for v_target, logits, values, a, boardstate, vectorstate in zip(buffer_v_target, logits, values, ba, buffer_boardstate_cpu, buffer_vectorstate_cpu):
        a = torch.tensor(a, dtype=torch.float32).to(device)
        values2, total_loss, c_loss, a_loss, entropy, l2_activity_loss = lnet.loss_func(boardstate, vectorstate, a, v_target, device)
        totalactorloss += a_loss + entropy + l2_activity_loss
        totalcriticloss += c_loss

        actorloss += a_loss + entropy + l2_activity_loss
        criticloss += c_loss
   
    actoropt.zero_grad()
        
    actorloss.backward()
    actorparameters = zip(lnet.actor_parameters, gnet.actor_parameters)
    for lp, gp in actorparameters:
        grad = lp.grad.to(global_device)
        gp._grad = grad
    actoropt.step() 

    criticopt.zero_grad()

    criticloss.backward()
    criticparameters = zip(lnet.critic_parameters, gnet.critic_parameters)
    for lp, gp in criticparameters:
        grad = lp.grad.to(global_device)
        gp._grad = grad

    criticopt.step()

    total_a_loss += a_loss
    total_c_loss += c_loss
    total_entropy += entropy
    total_l2_activity_loss += l2_activity_loss

    #del values2, total_loss, c_loss, a_loss, entropy, l2_activity_loss, criticloss, actorloss, actorparameters, criticparameters   

    #del actorparameters,criticparameters
    #del buffer_boardstate_cpu, buffer_vectorstate_cpu, buffer_v_target, a, boardstate, vectorstate

    #gc.collect()


    

        
#
    #values, loss, c_loss, a_loss, entropy, l2 = lnet.loss_func(
    #    v_wrap(np.vstack(buffer_boardstate_cpu)), 
    #    v_wrap(np.vstack(buffer_vectorstate_cpu)),
    #    v_wrap(np.array(ba, dtype=np.int64)) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
    #    v_wrap(np.array(buffer_v_target)[:, None]), device, total_step)
#
    ## calculate local gradients and push local parameters to global
    #actorloss = a_loss.mean() + entropy.mean() + l2.mean()
    #actoropt.zero_grad()
    #actorloss.backward()
#
    #for lp, gp in zip(lnet.actor_parameters, gnet.actor_parameters):
    #    grad = lp.grad.to(global_device)
    #    gp._grad = grad
    #actoropt.step() 
    #valueloss = 0
#
    #criticloss = c_loss.mean()
    #criticopt.zero_grad()
    #criticloss.backward()
#
    #for lp, gp in zip(lnet.critic_parameters, gnet.critic_parameters):
    #    grad = lp.grad.to(global_device)
    #    gp._grad = grad
    #criticopt.step()  
    
        
    
    valueloss = 0
    loss = 0
#
    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())
    return values, loss, total_c_loss, total_a_loss, total_entropy, total_l2_activity_loss, valueloss



def record(global_ep, global_ep_r, ep_r, res_queue, name):
    
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
        ""
    )

