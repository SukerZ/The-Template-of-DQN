'''定义超参数'''
max_steps = 100000
BUFFER_SIZE = 100000 
BATCH_SIZE = 32
GAMMA = 0.99
action_dim = 3
state_dim = 29
episode_count = 2000
max_steps = 100000
done = False
step = 0

if __name__ == "__main__":
    '''初始化：
       1、环境
       2、神经网络
       3、经验池
    '''
    np.random.seed(1337)
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    env.reset(relaunch=True)

    dqn = Brain(state_dim, action_dim)

    buff = ReplayBuffer(BUFFER_SIZE)

    for episode in range(episode_count):      #即for 𝑒𝑝𝑖𝑠𝑜𝑑𝑒=1 to 𝑇 do
        ob = env.reset(relaunch = True)       #每轮episode的初始化
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm) ).tolist()

        for j in range(max_steps):      #即for 𝑡=1 to 𝑇 do
            loss = 0
            a_t = dqn.explore_policy(s_t)    #探索策略选择动作a

            ob, r_t, done, info = env.step(a_t)   #执行动作a，从环境中获得<s,a,r,s'>
            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)).tolist()
            buff.add(s_t, a_t, r_t, s_t1, done)   #并存入经验池
            
            batch = buff.getBatch(BATCH_SIZE)     #从经验池中随机抽取训练样本
            states = [e[0] for e in batch]
            actions = [e[1] for e in batch]
            rewards = [e[2] for e in batch]
            new_states = [e[3] for e in batch]
            dones = [e[4] for e in batch]
            y_t = np.zeros((len(batch) ) )

            new_a = brain.target_policy(new_states) #目标策略选择argmaxQ(s',a')的a'
            target_q_values = brain.target_value(new_states, new_a)
            
            for j in range(len(batch) ):          #计算y_j的值
                if dones[j]:
                    y_t[j] = rewards[j]
                else:
                    y_t[j] = rewards[j] + GAMMA * target_q_values[j].item()
           
            pre = brain.predict(states, actions)
            loss += brain.train(pre, y_t)

            s_t = s_t1    #状态转移
            step += 1
            if done:
                break;

    env.reset()
    env.end()             #关闭环境
    print("Finish.")