from torch import nn
import torch
import numpy as np



def v_wrap(np_array, dtype=np.float32):
    torch.set_num_threads(1)
    if isinstance(np_array, torch.Tensor):
        np_array = np_array.cpu()
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, mean=0., std=1)
        nn.init.normal_(m.weight, mean=0., std=1)


def push_and_pull(opt, lnet, gnet, done, boardstate_, vectorstate_, buffer_boardstate, buffer_vectorstate, ba, br, gamma, device, global_device, total_step):
    torch.set_num_threads(1)
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(boardstate_, vectorstate_)[-1].data.cpu().numpy()[0, 0]


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

    values, loss, c_loss, a_loss, entropy, l2 = lnet.loss_func(
        v_wrap(np.vstack(buffer_boardstate_cpu)), 
        v_wrap(np.vstack(buffer_vectorstate_cpu)),
        v_wrap(np.array(ba, dtype=np.int64)) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]), device, total_step)

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        grad = lp.grad.to(global_device)
        gp._grad = grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())
    return values, loss, c_loss, a_loss, entropy, l2

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