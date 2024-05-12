from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import pdb
import pandas
from multiprocessing import Process, Lock, Manager, Pool
import sys

from config import config_global
sys.path.append("./util/")
from space import create_space_maestro, environment_maestro
import core_sac
from config_analyzer import config_self
from timer import timer
from recorder import recorder

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, size):
        self.obs_buf = np.zeros(core_sac.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core_sac.combined_shape(size, obs_dim), dtype=np.float32)
        #self.act_buf = np.zeros(core_sac.combined_shape(size, act_dim), dtype=np.float32)
        self.act_buf = [0]*size
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.step_buf = np.zeros(size, dtype=np.int32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, step):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.step_buf[self.ptr] = step
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        act_buf_temp = list()
        for idx in idxs:
            act_buf_temp.append(torch.as_tensor(self.act_buf[idx], dtype=torch.float32))
        obs_buf_temp = torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32)
        obs2_buf_temp = torch.as_tensor(self.obs2_buf[idxs], dtype=torch.float32)
        rew_buf_temp = torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32)
        done_buf_temp = torch.as_tensor(self.done_buf[idxs], dtype=torch.float32)
        step_buf_temp = torch.as_tensor(self.step_buf[idxs], dtype=torch.float32)
        batch = dict(obs=obs_buf_temp,
        			 obs2=obs2_buf_temp,
        			 act=act_buf_temp,
        			 rew=rew_buf_temp,
        			 done=done_buf_temp,
        			 step=step_buf_temp)
        '''
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     #act=self.act_buf[idxs],
                     act = act_buf_temp,
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     step=self.step_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
        '''
        return {k:v for k,v in batch.items()}



def sac(iindex,  actor_critic=core_sac.MLPActorCritic_DSE, 
        batch_size=20, replay_size=int(1e4), max_ep_len=100000, 
        gamma=0.999, polyak=0.995, lr=1e-3, alpha=0.2, 
        logger_kwargs=dict(), save_freq=1):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    algo="SAC"
    seed = iindex * 10000
    atype = int(iindex / 10)
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = config_self(iindex)
    config.config_check()
    env = environment_maestro(
        algo= algo,
        iindex = iindex,
        config = config,
        test = True,
        delay_reward = True
    )

    test_env = environment_maestro(
        algo= algo,
        iindex = iindex,
        config = config,
        test = True,
        delay_reward = True
    )

    steps_per_epoch = env.design_space_dimension
    if(env.delay_reward):
        epochs = config.period
    else:
        epochs = int(config.period/steps_per_epoch)
    print(f"delay_reward:{env.delay_reward}, epochs:{epochs}")
    start_epochs = 1
    total_steps = steps_per_epoch * (epochs + start_epochs)
    start_steps = steps_per_epoch * start_epochs
    update_after = start_steps
    update_every = steps_per_epoch * 1
    update_times = 1
    num_test_episodes= 1

    #obs_dim = env.design_space_dimension
    obs_dim = env.const_lenth + env.dynamic_lenth
    # Create actor-critic module and target networks
    ac = actor_critic(
        #obs_dim = env.design_space_dimension,
        obs_dim = obs_dim,
        act_dim_list = env.action_dimension_list,
        act_limit_list = env.action_limit_list
    ) 
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, size=replay_size)

    tm = timer()

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d, n = data['obs'], data['act'], data['rew'], data['obs2'], data['done'], data['step']
        z = (n+1)%env.design_space_dimension

        q1 = ac.q1(o,a,n)
        q2 = ac.q2(o,a,n)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2, z)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2, z)
            q2_pi_targ = ac_targ.q2(o2, a2, z)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o,n = data['obs'], data['step']
        pi, logp_pi = ac.pi(o,n)
        q1_pi = ac.q1(o, pi, n)
        q2_pi = ac.q2(o, pi, n)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    #logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()
        #print(f"loss_pi:{loss_pi}")

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, n, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), n, deterministic)

    def test_agent(epoch):
        r_avg = 0
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                n = ep_len % env.design_space_dimension
                o, r, d, _ = test_env.step(n, get_action(o, n, True))
                ep_ret += r
                ep_len += 1
            r_avg += ep_ret
        r_avg = r_avg/num_test_episodes
        print(f"epoch = {epoch}, test:reward = {r_avg}, best_result = {env.best_result}", end='\r')
            #logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    tm.start("all")
    for t in range(total_steps):
        n = t % env.design_space_dimension
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            env.test = False
            a = get_action(o,n)
        else:
            a = env.sample(n)

        # Step the env
        tm.start("eva")
        o2, r, d, _ = env.step(n, a)
        tm.end("eva")
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d, n)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0


        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_times):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
    tm.end("all")
    return env.best_objectvalue_list, env.multiobjecvalue_list, tm

def run(args):
    iindex, objective_record, timecost_record, multiobjective_record = args
    print(f"%%%%TEST{iindex} START%%%%")
    best_objectvalue_list, multiobjecvalue_list, tm = sac(iindex)

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
    algoname = "SAC_MOO"
    use_multiprocess = True
    global_config = config_global()
    TEST_BOUND = global_config.TEST_BOUND
    #PROCESS_NUM = global_config.PROCESS_NUM
    PROCESS_NUM = 2
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
