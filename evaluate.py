from agent.agent import Agents
from common.rollout import Evaluator
from common.arguments import get_evaluate_args
import time


if __name__ == '__main__':
    Psuccess=[]
    Tavg=[]
    args, ENV = get_evaluate_args()
    for i in range(1):
        # ----一次运行FF
        start=time.time()
        env = ENV(args.width, args.length, args.drop_num, n_blocks=args.block_num, fov=args.fov, stall=args.stall, show=args.show, savemp4=args.show_save)
        args.__dict__.update(env.get_env_info())
        evaluator = Evaluator(env, Agents(args), args.episode_limit)
        average_episode_rewards, average_episode_steps, _, success_rate = evaluator.evaluate(args.evaluate_task)
        Psuccess.append(success_rate)
        Tavg.append(average_episode_steps)
        print('time:',time.time()-start)
        print('The average total_rewards of {} is  {}'.format(
            args.alg, average_episode_rewards))
        print('The average total_steps is: {}'.format(
            average_episode_steps))
        print('The successful rate is: {}'.format(success_rate))
    # ----
    # if args.evaluate:
    # np.save('Psuccess_{}_{}_{}'.format(args.width, args.drop_num, args.fov), Psuccess)
    # np.save('Tavg_{}_{}_{}'.format(args.width, args.drop_num, args.fov), Tavg)

