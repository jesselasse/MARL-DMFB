from common.arguments import process_assay_args
from munch import DefaultMunch
from agent.agent import Agents
from env.input import DAG, PCR
import numpy as np




class Process_assay:
    def __init__(self,env, agents):
        self.agents = agents
        self.n_actions = (agents[0].n_actions, agents[1].n_actions, agents[2].n_actions)
        self.env=env

    def process_assay(self):
        max_step = 100000
        # start process
        terminated = False
        step = 0
        obs = self.env.getObs()
        while not terminated and step < max_step:
            actions, actions_onehot = [], []
            avail_actions = self.env.routing_manager.get_avail_action(self.n_actions[0])
            for i in range(self.env.n_agents):
                d = self.env.routing_manager.droplets[i]
                tp=d.type
                action, hidden_state = self.agents[tp].choose_action(
                    obs[i], avail_actions[i], 0, d.last_action, d.hidden_state)
                action_onehot = np.zeros(self.n_actions[tp])
                action_onehot[action] = 1
                actions.append(int(action))
                actions_onehot.append(action_onehot)
                d.hidden_state = hidden_state
                d.last_action = action_onehot
            obs, rewards, terminated, info = self.env.step(actions)
            self.env.render()
            step += 1



if __name__ == '__main__':
    args, ENV, Chip, Manager, netdata = process_assay_args()

    # inputs
    G = DAG(PCR())
    chip = Chip(args.width, args.length)
    chip.create_dispense(G.dispensers.__len__() + 1)
    G.assign_ports(chip.ports)
    assayManager=Manager(chip, G, fov=[nd['fov'] for nd in netdata]+[9], stall=args.stall)
    env = ENV(assayManager,
              show=args.show, savemp4=args.show_save)
    args.__dict__.update(env.get_env_info())
    args.n_agents=8
    load_model_name=['4_28_',args.load_model_name,'4_28_']
    obs_shape=args.obs_shape

  #  args.load_model_name = ['3_', args.load_model_name, '4_28_']
    agents=[]
    for i in range(2):
        nargs = DefaultMunch.fromDict(netdata[i])
        nargs.load_model_name = load_model_name[i]
        nargs.obs_shape = obs_shape[i]
        nargs.n_actions = args.n_actions
        nargs.task = i
        agents.append(Agents(args, nargs, task=i))
    args.load_model = False
    agents.append(Agents(args, nargs, task=2))
    process=Process_assay(env, agents)
    process.process_assay()

    #更新 CL
