from torch import nn
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import gc 



def v_wrap(np_array, dtype=np.float32):
    """
    Converts a NumPy array to a PyTorch tensor.
    """
    torch.set_num_threads(1)
    
    # Convert the input array to CPU if it is a torch.Tensor
    if isinstance(np_array, torch.Tensor):
        np_array = np_array.cpu()
    
    # Convert the data type of the array if it is not the desired dtype
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    
    # Convert the NumPy array to a PyTorch tensor and return it
    return torch.from_numpy(np_array)


def init_weights(m):
    """
    Initializes the weights and biases of a PyTorch module.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # Initialize weights using the Kaiming normal initialization method
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            # Initialize biases to zero
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # Initialize batch normalization weights to 1 and biases to 0
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def push_and_pull(actoropt, criticopt, lnet, gnet, done, boardstate_, vectorstate_, buffer_boardstate, buffer_vectorstate, ba, br, gamma, device, global_device, logits, values):
    """
    Function for updating the actor and critic networks using the asynchronous advantage actor-critic (A3C) algorithm.
    """
    torch.set_num_threads(1)

    # Calculate the value of the next state
    if done:
        v_s_ = 0.
    else:
        v_s_ = lnet.forward(boardstate_, vectorstate_)[-1].data.cpu().numpy()[0, 0]

    buffer_v_target = []

    # Calculate the actual value for each state in the buffer
    if len(br) == 1:
        buffer_v_target.append(br[0] + gamma * v_s_)
    else:
        for r in br[::-1]:
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

    # Calculate the loss for each prediction throughout the game
    for v_target, logits, values, a, boardstate, vectorstate in zip(buffer_v_target, logits, values, ba, buffer_boardstate_cpu, buffer_vectorstate_cpu):
        a = torch.tensor(a, dtype=torch.float32).to(device)
        _, _, c_loss, a_loss, entropy, l2_activity_loss = lnet.loss_func(boardstate, vectorstate, a, v_target, device)
        totalactorloss += a_loss + entropy + l2_activity_loss
        totalcriticloss += c_loss

        actorloss += a_loss + entropy + l2_activity_loss
        criticloss += c_loss

    actoropt.zero_grad()
    actorloss.backward()

    # Update the global actor network
    actorparameters = zip(lnet.actor_parameters, gnet.actor_parameters)
    for lp, gp in actorparameters:
        grad = lp.grad.to(global_device)
        gp._grad = grad
    actoropt.step()

    criticopt.zero_grad()
    criticloss.backward()

    # Update the global critic network
    criticparameters = zip(lnet.critic_parameters, gnet.critic_parameters)
    for lp, gp in criticparameters:
        grad = lp.grad.to(global_device)
        gp._grad = grad
    criticopt.step()

    total_a_loss += a_loss
    total_c_loss += c_loss
    total_entropy += entropy
    total_l2_activity_loss += l2_activity_loss
    valueloss = 0
    loss = 0

    lnet.load_state_dict(gnet.state_dict())

    return values, loss, total_c_loss, total_a_loss, total_entropy, total_l2_activity_loss, valueloss
def record(global_ep, global_ep_r, ep_r, res_queue, name):
    """
    Function for recording the results of the training process.
    """
    
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

