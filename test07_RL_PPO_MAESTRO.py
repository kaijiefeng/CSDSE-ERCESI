import numpy as np
import torch
from torch.optim import Adam
import time
import pdb
import pandas
import os
from multiprocessing import Process, Lock, Manager, Pool
import sys

from config import config_global
sys.path.append("./util/")
from space import create_space_maestro, environment_maestro
import core_ppo
from config_analyzer import config_self
from timer import timer
from recorder import recorder

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core_ppo.combined_shape(size, obs_dim), dtype=np.float32)
        #***********************************Patch***********************************#
        self.act_buf = np.zeros(size, dtype=np.float32)
        #***********************************Patch***********************************#
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        #***********************************Patch***********************************#
        self.step_buf = np.zeros(size, dtype=np.float32)
        #***********************************Patch***********************************#
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, step):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        #***********************************Patch***********************************#
        self.step_buf[self.ptr] = step
        #***********************************Patch***********************************#
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core_ppo.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core_ppo.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick

        #***********************************Patch***********************************#
        #adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        #self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        #***********************************Patch***********************************#

        #***********************************Patch***********************************#
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, step=self.step_buf)
        #***********************************Patch***********************************#
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



def ppo(iindex, actor_critic=core_ppo.MLPActorCritic_DSE,
        batch_size = 5, max_ep_len=100000,
        gamma=0.999, clip_ratio=0.2, pi_lr=0.001, vf_lr=0.01, 
        train_pi_iters=1, train_v_iters=10, lam=0.97, target_kl=0.01, 
        logger_kwargs=dict(), save_freq=10):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    algo="PPO"
    seed = iindex * 10000
    torch.manual_seed(seed)
    np.random.seed(seed)
    #### assign DSE config
    config = config_self(iindex)
    config.config_check()
    env = environment_maestro(
        algo= algo,
        iindex = iindex,
        config = config,      
    )
    #obs_dim = env.design_space_dimension
    obs_dim = env.const_lenth + env.dynamic_lenth
    act_dim_list = env.action_dimension_list
    ac = actor_critic(
        obs_dim = obs_dim,
        act_dim_list = act_dim_list
    )
    steps_per_epoch = env.design_space_dimension
    local_steps_per_epoch = int(steps_per_epoch) * batch_size
    epochs = int(config.period/batch_size)
    tm = timer()

    # Set up experience buffer
    buf = PPOBuffer(obs_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old, step = data['obs'], data['act'], data['adv'], data['logp'], data['step']

        # Policy loss
        pi, logp = ac.pi(obs, step, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        #***********************************Patch***********************************#
        ent = 0
        for pi_i in pi:
            ent += pi_i.entropy().item()
        ent = ent / len(pi)
        #***********************************Patch***********************************#
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                break
            loss_pi.backward()
            pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()
        #sprint(f"critic_loss = {loss_v}")

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    r_avg = 0

    # Main loop: collect experience in env and update/log each epoch
    tm.start("all")
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            n = t % env.design_space_dimension
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32), n)

            tm.start("eva")
            next_o, r, d, _ = env.step(n, a)
            tm.end("eva")
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp, n)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32), n)
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    pass
                r_avg += ep_ret

                o, ep_ret, ep_len = env.reset(), 0, 0
                #pdb.set_trace()
        r_avg = 0

        # Perform PPO update!
        update()
    tm.end("all")

    return env.best_objectvalue_list, env.multiobjecvalue_list, tm

def run(args):
    iindex, objective_record, timecost_record, multiobjective_record = args
    print(f"%%%%TEST{iindex} START%%%%")
    best_objectvalue_list, multiobjecvalue_list, tm = ppo(iindex)

    timecost_list = tm.get_list("all")
    evacost = tm.get_sum("eva")
    timecost_list.append(evacost)
    
    best_objectvalue_list.append(iindex)
    timecost_list.append(iindex)
    multiobjecvalue_list.append([iindex, iindex])
    objective_record.append(best_objectvalue_list)
    timecost_record.append(timecost_list)
    multiobjective_record.append(multiobjecvalue_list)

if __name__ == '__main__':
    algoname = "PPO_MOO"
    use_multiprocess = True
    global_config = config_global()
    TEST_BOUND = global_config.TEST_BOUND
    PROCESS_NUM = global_config.PROCESS_NUM
    SCEN_TYPE = global_config.SCEN_TYPE
    SCEN_NUM = global_config.SCEN_NUM
    PASS = global_config.PASS

    args_list = list()
    objective_record = Manager().list()
    timecost_record = Manager().list()
    multiobjective_record = Manager().list()

    if(use_multiprocess):
        args_list = list()
        for iindex in range(TEST_BOUND):
            if(iindex in PASS): continue
            args_list.append((iindex, objective_record, timecost_record, multiobjective_record))
        pool = Pool(PROCESS_NUM)
        pool.map(run, args_list)
        pool.close()
        pool.join()
    else:
        for iindex in range(TEST_BOUND):
            if(iindex in PASS): continue
            run((iindex, objective_record, timecost_record))

    recorder(algoname, global_config, objective_record, timecost_record, multiobjective_record)
