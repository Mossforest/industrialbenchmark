# coding=utf-8
from __future__ import division
import numpy as np
from industrial_benchmark_python.goldstone.environment import environment as GoldstoneEnvironment
from industrial_benchmark_python.EffectiveAction import EffectiveAction
from collections import OrderedDict
'''
The MIT License (MIT)

Copyright 2017 Siemens AG

Author: Stefan Depeweg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

class IDS(object):
    '''
    Lightweight python implementation of the industrial benchmark
    Uses the same standard settings as in src/main/ressources/simTest.properties 
    of java implementation
    '''

    def __init__(self,p=50,stationary_p=True, inital_seed=None):
        '''
        p sets the setpoint hyperparameter (between 1-100) which will
        affect the dynamics and stochasticity.

        stationary_p = False will make the setpoint vary over time. This
        will make the system more non-stationary.
        '''

        # fix seed only for testing
        #if inital_seed != None:
        np.random.seed(inital_seed)

        # constants
        self.maxRequiredStep = np.sin(15./180.*np.pi)
        self.gsBound = 1.5
        self.gsSetPointDependency = 0.02
        self.gsScale = 2.*self.gsBound + 100.*self.gsSetPointDependency # scaling factor for shift

        # cost/reward weighting constants
        self.CRF = 3.
        self.CRC = 1.
        self.CRGS =  25.

        self.stationary_p = stationary_p
        self.gsEnvironment = GoldstoneEnvironment(24, self.maxRequiredStep, self.maxRequiredStep/2.0)

        self.state = OrderedDict()

        self.state['o'] = np.zeros(10) #  operational cost buffer
        self.state['coc'] = 0 # current operational cost
        self.state['fb'] = 0.  # basic fatigue: without bifurcation aspects
        self.state['oc'] = 0 # current operational cost conv
        self.state['hg'] = 0. # hidden gain
        self.state['hv'] = 0. # hidden velocity
        self.state['he'] = 0. # hidden/ effective shift

        # goldstone variables
        self.state['gs_domain'] = self.gsEnvironment._dynamics.Domain.positive.value # miscalibration domain
        self.state['gs_sys_response'] = self.gsEnvironment._dynamics.System_Response.advantageous.value # miscalibration System Response
        self.state['gs_phi_idx'] = 0 # miscalibration Phi_idx/ direction
        self.state['ge'] = 0. # effective action gain beta
        self.state['ve'] = 0. # effective action velocity alpha
        self.state['MC'] = 0. # Miscalibration

        # observables
        self.observable_keys = ['p','v','g','h','f','c','cost','reward']
        self.state['p'] = p  # SetPoint
        self.state['v'] = 50. # Velocity
        self.state['g'] = 50.  # Gain
        self.state['h'] = 50.  # Shift
        self.state['f'] = 0.  # fatigue
        self.state['c'] = 0. # consumption
        self.state['cost'] = 0. #  signal/ total
        self.state['reward'] = 0. # reward

        self.init = True
        self.defineNewSequence()
        self.step(np.zeros(3))

    def visibleState(self):
        return np.concatenate([np.array(self.state[k]).ravel() for k in self.observable_keys])

    def markovState(self):
        return np.concatenate([np.array(self.state[k]).ravel() for k in self.state.keys()])

    def step(self,delta):
        self.updateSetPoint()
        self.addAction(delta)
        self.updateFatigue()
        self.updateCurrentOperationalCost()
        self.updateOperationalCostConvolution()
        self.updateGS()
        self.updateOperationalCosts()
        self.updateCost() 

    def updateSetPoint(self):
        if self.stationary_p == True:
            return
        else:
            if self._p_step == self._p_steps:
                self.defineNewSequence()

            new_p = self.state['p'] + self._p_ch
            if new_p > 100 or new_p < 0:

                if np.random.rand() > 0.5:
                    self._p_ch *= -1

            new_p = np.clip(new_p, 0, 100)

            self.state['p'] = new_p

            self._p_step += 1

    def addAction(self,delta):
        delta = np.clip(delta,-1,1)
        self.state['v'] = np.clip(self.state['v'] + delta[0], 0., 100.)
        self.state['g'] = np.clip(self.state['g'] + 10 * delta[1], 0., 100.)
        self.state['h'] = np.clip(self.state['h'] + ((self.maxRequiredStep / 0.9) * 100. / self.gsScale) * delta[2], 0.,100.)
        self.state['he'] = np.clip(self.gsScale*self.state['h']/100. - self.gsSetPointDependency*self.state['p'] - self.gsBound,-self.gsBound,self.gsBound)


    def updateFatigue(self):
        expLambda = 0.1
        actionTolerance = 0.05
        fatigueAmplification = 1.1    
        fatigueAmplificationMax = 5.0
        fatigueAmplificationStart = 1.2 

        dyn = 0.0
        velocity = self.state['v']
        gain = self.state['g']
        setpoint = self.state['p']

        hidden_gain = self.state['hg']
        hidden_velocity = self.state['hv']

        effAct =  EffectiveAction(velocity,gain,setpoint)
        effAct_velocity = effAct.getEffectiveVelocity()
        effAct_gain = effAct.getEffectiveGain()

        self.state['ge'] = effAct_gain
        self.state['ve'] = effAct_velocity

        noise_e_g = np.random.exponential(expLambda)
        noise_e_v = np.random.exponential(expLambda)
        noise_u_g = np.random.rand()
        noise_u_v = np.random.rand()
        
        noise_b_g = np.float64(np.random.binomial(1, np.clip(effAct_gain,0.001, 0.999)))
        noise_b_v = np.float64(np.random.binomial(1, np.clip(effAct_velocity,0.001, 0.999)))

        noise_gain = 2.0 * (1.0/(1.0+np.exp(-noise_e_g)) - 0.5)
        noise_velocity = 2.0 * (1.0/(1.0+np.exp(-noise_e_v)) - 0.5)

        noise_gain += (1-noise_gain) * noise_u_g * noise_b_g * effAct_gain
        noise_velocity += (1-noise_velocity) * noise_u_v * noise_b_v * effAct_velocity

        if effAct_gain <= actionTolerance:
            hidden_gain = effAct_gain
        elif hidden_gain >= fatigueAmplificationStart:
            hidden_gain = np.minimum(fatigueAmplificationMax, fatigueAmplification * hidden_gain)
        else:
            hidden_gain = 0.9 * hidden_gain + noise_gain / 3.

        if effAct_velocity <= actionTolerance:
            hidden_velocity = effAct_velocity
        elif hidden_velocity >= fatigueAmplificationStart:
            hidden_velocity = np.minimum(fatigueAmplificationMax, fatigueAmplification * hidden_velocity)
        else:
            hidden_velocity = 0.9 * hidden_velocity + noise_velocity / 3.

        if np.maximum(hidden_velocity, hidden_gain) == fatigueAmplificationMax:
            alpha = 1.0 / (1.0 + np.exp(-np.random.normal(2.4, 0.4)))
        else:
            alpha = np.maximum(noise_velocity, noise_gain)

        fb = np.maximum(0,((30000. / ((5*velocity) + 100)) - 0.01 * (gain**2)))
        self.state['hv'] = hidden_velocity
        self.state['hg'] = hidden_gain
        self.state['f'] = (fb*(1+2*alpha)) / 3.
        self.state['fb'] = fb

    def updateCurrentOperationalCost(self):
        CostSetPoint = 2.
        CostVelocity = 4.
        CostGain = 2.5

        gain = self.state['g']
        velocity = self.state['v']
        setpoint = self.state['p']

        costs = CostSetPoint * setpoint + CostGain * gain + CostVelocity * velocity
        o = np.exp(costs / 100.)
        self.state['coc'] = o

        if self.init == True:
            self.state['o'] += o
            self.init = False
        else:
            self.state['o'][:-1] = self.state['o'][1:]
            self.state['o'][-1] = o

    def updateOperationalCostConvolution(self):
        ConvArray=np.array([0.11111,0.22222,0.33333,0.22222,0.11111,0.,0.,0.,0.,0.])
        self.state['oc'] = np.dot(self.state['o'],ConvArray)

    def updateGS(self):
        setpoint = self.state['p']
        shift = self.state['h']
        effective_shift = self.state['he']

        domain = self.state['gs_domain']
        phi_idx = self.state['gs_phi_idx']
        system_response = self.state['gs_sys_response']

        reward, domain, phi_idx, system_response = self.gsEnvironment.state_transition(self.gsEnvironment._dynamics.Domain(domain), phi_idx, self.gsEnvironment._dynamics.System_Response(system_response), effective_shift)
        self.state['MC'] = -reward
        self.state['gs_domain'] = domain.value
        self.state['gs_sys_response'] = system_response.value
        self.state['gs_phi_idx'] = phi_idx

    def updateOperationalCosts(self):
        rGS = self.state['MC']
        eNewHidden = self.state['oc'] - (self.CRGS * (rGS - 1.0))
        operationalcosts = eNewHidden - np.random.randn()*(1+0.005*eNewHidden)
        self.state['c'] = operationalcosts

    def updateCost(self):
        fatigue = self.state['f']
        consumption = self.state['c']
        cost = self.CRF * fatigue + self.CRC * consumption

        self.state['cost'] =  cost
        self.state['reward'] = -cost

    def defineNewSequence(self):
        length = np.random.randint(1,100)
        self._p_steps = length
        self._p_step = 0
        p_ch = 2 * np.random.rand() -1
        if np.random.rand() < 0.1:
            p_ch *= 0.
        self._p_ch =  p_ch
    
    def save_state(self):
        """
        保存环境的完整状态
        """
        state_dict = {
            # 保存主状态字典的深拷贝
            'state': self.state.copy(),
            
            # 保存setpoint相关变量
            'p_step': getattr(self, '_p_step', None),
            'p_steps': getattr(self, '_p_steps', None),
            'p_ch': getattr(self, '_p_ch', None),
            
            # 保存随机数生成器状态
            'random_state': np.random.get_state(),
            
            # 保存Goldstone环境的dynamics状态
            'gs_dynamics_state': self.gsEnvironment._dynamics.save_state()
        }
        return state_dict

    def load_state(self, state_dict):
        """
        从保存的状态字典中恢复环境状态
        """
        # 恢复主状态字典
        self.state = state_dict['state'].copy()
        
        # 恢复setpoint相关变量
        if state_dict['p_step'] is not None:
            self._p_step = state_dict['p_step']
            self._p_steps = state_dict['p_steps']
            self._p_ch = state_dict['p_ch']
        
        # 恢复随机数生成器状态
        np.random.set_state(state_dict['random_state'])
        
        # 恢复Goldstone环境的dynamics状态
        self.gsEnvironment._dynamics.load_state(state_dict['gs_dynamics_state'])

if __name__ == "__main__":
    """
    验证环境状态的保存和加载功能
    """
    def compare_states(state1, state2):
        """比较两个状态字典是否相同"""
        if set(state1.keys()) != set(state2.keys()):
            print("状态键不匹配")
            return False
            
        for key in state1:
            if isinstance(state1[key], np.ndarray):
                if not np.array_equal(state1[key], state2[key]):
                    print(f"数组不匹配: {key}")
                    return False
            elif isinstance(state1[key], dict):
                if not compare_states(state1[key], state2[key]):
                    print(f"子字典不匹配: {key}")
                    return False
            elif isinstance(state1[key], tuple):
                # 处理tuple类型（比如random_state）
                if not all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b 
                          for a, b in zip(state1[key], state2[key])):
                    print(f"元组不匹配: {key}")
                    return False
            else:
                try:
                    if state1[key] != state2[key]:
                        print(f"值不匹配: {key}")
                        print(f"state1[{key}] = {state1[key]}")
                        print(f"state2[{key}] = {state2[key]}")
                        return False
                except ValueError:
                    print(f"无法比较的值类型: {key}")
                    print(f"state1[{key}] 类型: {type(state1[key])}")
                    print(f"state2[{key}] 类型: {type(state2[key])}")
                    return False
        return True

    # 测试1：基本状态保存和加载
    print("测试1：基本状态保存和加载")
    env = IDS(p=50)
    state1 = env.save_state()
    env.load_state(state1)
    state2 = env.save_state()
    print("基本保存和加载测试通过：", compare_states(state1, state2))

    # 测试2：执行动作后的状态保存和加载
    print("\n测试2：执行动作后的状态保存和加载")
    env = IDS(p=50)
    env.step(np.array([0.5, 0.3, -0.2]))
    state1 = env.save_state()
    
    # 保存一些关键值用于验证
    cost1 = env.state['cost']
    reward1 = -cost1
    
    env.step(np.array([-0.1, 0.4, 0.1]))  # 执行另一个动作改变状态
    env.load_state(state1)  # 加载回之前的状态
    
    cost2 = env.state['cost']
    reward2 = -cost2
    state2 = env.save_state()
    
    print("状态匹配：", compare_states(state1, state2))
    print("成本匹配：", cost1 == cost2)
    print("奖励匹配：", reward1 == reward2)

    # 测试3：多次保存加载的一致性
    print("\n测试3：多次保存加载的一致性")
    env = IDS(p=50)
    states = []
    for _ in range(5):
        env.step(np.random.uniform(-1, 1, 3))
        state = env.save_state()
        states.append(state)
        
    # 逆序加载状态并验证
    for i in range(len(states)-1, -1, -1):
        env.load_state(states[i])
        current_state = env.save_state()
        print(f"第{i+1}个状态匹配：", compare_states(states[i], current_state))

    # 测试4：随机性测试
    print("\n测试4：随机性测试")
    env = IDS(p=50)
    state1 = env.save_state()
    
    # 执行相同的动作两次，检查是否得到不同的结果
    action = np.array([0.5, 0.3, -0.2])
    env.step(action)
    result1 = env.state['cost']
    
    env.load_state(state1)
    env.step(action)
    result2 = env.state['cost']
    
    print("随机性保持（结果应该不同）：", result1 != result2)