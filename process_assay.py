from common.arguments import common_args,evaluate_args,process_assay_args
from agent.agent import Agents
from env.input import DAG, PCR, in_vitro
import numpy as np

def get_assay_args():
    parser = common_args()
    parser = evaluate_args(parser)
    parser.add_argument('--load_model_name2', type=str, default='', help=' we can choose the file name, 例：1_7500 '
                             '会找到指定1_7500_rnn_net_params.pkl文件')
    parser.add_argument('--assay', type=int, default=0, choices=[0, 1, 2], help='assay name, 0:PCR; 1: invitro')
    parser.add_argument('--invitroParas', type=int, default=0, choices=[0, 1, 2], help='invitro parameters, 0: (3,3); 1: (3,4); 2: (4,4)')
    args = parser.parse_args()
    return process_assay_args(args)


class Process_assay:
    def __init__(self,env, agents):
        self.agents = agents
        self.n_actions = (agents[0].n_actions, agents[1].n_actions, agents[2].n_actions)
        self.env=env

    def process_assay(self):
        max_step = 1000
        # start process
        terminated = False
        step = 0
        success = 0
        constraints = 0
        obs = self.env.reset()
        while not terminated and step < max_step:
            actions, actions_onehot = [], []
            avail_actions = self.env.routing_manager.get_avail_action(self.n_actions[0])
            for i in range(self.env.n_agents):
                d = self.env.routing_manager.droplets[i]
                tp = d.type
                action, hidden_state = self.agents[tp].choose_action(
                    obs[i], avail_actions[i], 0, d.last_action, d.hidden_state)
                actions.append(int(action))
                d.hidden_state = hidden_state
            obs, rewards, terminated, info = self.env.step(actions)
            success += info['success']
            constraints += info['constraints']
            self.env.render()
            step += 1
        return step, success, constraints

size={'FC4': 5, 'CRatt': (2,9,9,5), 'crnn': (2,9,9,2,164)}

if __name__ == '__main__':
    args, ENV, Chip, Manager= get_assay_args()
    para = None
    D = [PCR, in_vitro][args.assay]
    if args.assay == 1:
        para = [(3,3),(3,4),(4,4)][args.invitroParas]
    # inputs
    G = DAG(D(para))
    chip = Chip(args.width, args.length)
    #
    # chip.create_dispense(8 + 1)
    # G.assign_ports(chip.ports, porttype[D])
    assayManager=Manager(chip, G, fov=[nd.fov for nd in args.netdata]+[9], oc=args.oc, stall=args.stall)
    env = ENV(assayManager,
              show=args.show, savemp4=args.show_save)
    args.__dict__.update(env.get_env_info())
    # args.obs_shape = size[args.net]
    args.n_agents=8
    obs_shape=args.obs_shape

  #  args.load_model_name = ['3_', args.load_model_name, '4_28_']
    agents=[]
    for i in range(2):
        agents.append(Agents(args, task=i))
    args.load_model = False
    agents.append(Agents(args, task=2))
    process=Process_assay(env, agents)
    n=args.evaluate_task
    steps=np.zeros(n)
    successes=np.zeros(n)
    constraints=np.zeros(n)
    records = [steps,successes,constraints]
    for i in range(n):
        record =process.process_assay()
        for r in range(3):
            records[r][i]=record[r]
    # steps,successes,constraints=records
    print('step:',steps,'success:',successes,'constraint:',constraints)
    print('mean step:',steps.mean(),'mean success:',successes.mean(),'mean constraint:',constraints.mean())
    #更新 CL
