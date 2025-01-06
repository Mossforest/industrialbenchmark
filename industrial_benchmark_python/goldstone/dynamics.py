# coding: utf8
import numpy as np
from numpy import pi, sign
from industrial_benchmark_python.goldstone import reward_function
from enum import Enum
'''
The MIT License (MIT)

Copyright 2017 Siemens AG

Author: Judith Mosandl

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

class dynamics:
    class Domain(Enum):
        negative = -1
        positive = +1

    class System_Response(Enum):
        advantageous = +1
        disadvantageous = -1

    def __init__(self, number_steps, max_required_step, safe_zone):
        self._safe_zone = self._check_safe_zone(safe_zone)
        self._strongest_penality_abs_idx = self.compute_strongest_penalty_absIdx(number_steps)
        self._penalty_functions_array = self._define_reward_functions(number_steps, max_required_step)

    def _check_safe_zone(self, safe_zone):
        if (safe_zone < 0):
            raise ValueError('safe_zone must be non-negative')
        return safe_zone

    def reset(self):
        return self.Domain.positive, 0, self.System_Response.advantageous

    def reward(self, Phi_idx, position):
        return self.get_penalty_function(Phi_idx).reward(position)

    def state_transition(self, domain, phi_idx, system_response, position):

        old_domain = domain

        # (0) compute new domain
        domain = self._compute_domain(old_domain, position)

        # (1) if domain change: system_response <- advantageous
        if domain != old_domain:
            system_response = self.System_Response.advantageous

        # (2) compute & apply turn direction
        phi_idx += self._compute_angular_step(domain, phi_idx, system_response, position)

        # (3) Update system response if necessary
        system_response = self._updated_system_response(phi_idx, system_response)

        # (4) apply symmetry
        phi_idx = self._apply_symmetry(phi_idx)

        # (5) if self._Phi_idx == 0: reset internal state
        if (phi_idx == 0) and (abs(position) <= self._safe_zone):
            domain, phi_idx, system_response = self.reset()

        return domain, phi_idx, system_response

    def _compute_domain(self, domain, position):
        #compute the new domain of control action
        if abs(position) <= self._safe_zone:
            return domain
        else:
            return self.Domain(sign(position))

    def _compute_angular_step(self, domain, phi_idx, system_response, position):
        # cool down: when position close to zero
        if abs(position) <= self._safe_zone:  # cool down
            return -sign(phi_idx)

        if phi_idx == -domain.value * self._strongest_penality_abs_idx:
            return 0
        return system_response.value * sign(position)

    def _updated_system_response(self, phi_idx, system_response):
        if abs(phi_idx) >= self._strongest_penality_abs_idx:
            return self.System_Response.disadvantageous
        else:
            return system_response

    def _apply_symmetry(self, phi_idx):
        if abs(phi_idx) < self._strongest_penality_abs_idx:
            return phi_idx

        phi_idx = (phi_idx + (4 * self._strongest_penality_abs_idx)) % (4 * self._strongest_penality_abs_idx)
        phi_idx = 2 * self._strongest_penality_abs_idx - phi_idx
        return phi_idx

    def get_penalty_function(self, phi_idx):
        idx = int(self._strongest_penality_abs_idx + phi_idx)
        if idx < 0:
            idx = idx + len(self._penalty_functions_array)
        return self._penalty_functions_array[idx]

    def _define_reward_functions(self, number_steps, max_required_step):
        k = self._strongest_penality_abs_idx
        angle_gid = np.arange(-k, k + 1) * 2 * pi / number_steps
        reward_functions = [reward_function.reward_function(Phi, max_required_step) for Phi in angle_gid]

        self._penalty_functions_array = np.array(reward_functions)
        return self._penalty_functions_array

    def compute_strongest_penalty_absIdx(self, number_steps):
        if (number_steps < 1) or (number_steps % 4 != 0):
            raise ValueError('number_steps must be positive and integer multiple of 4')

        _strongest_penality_abs_idx = number_steps // 4
        return _strongest_penality_abs_idx

    def save_state(self):
        """
        保存dynamics的完整状态
        """
        state_dict = {
            # 初始化参数
            'number_steps': getattr(self, 'number_steps', None),
            'max_required_step': self.maxRequiredStep if hasattr(self, 'maxRequiredStep') else None,
            'safe_zone': self._safe_zone,
            
            # 计算的中间值
            'strongest_penalty_abs_idx': self._strongest_penality_abs_idx,
            
            # 奖励函数相关
            'penalty_functions_array': self._penalty_functions_array,
            
            # 其他可能的状态变量
            'last_reward': getattr(self, '_last_reward', None),
            'last_position': getattr(self, '_last_position', None)
        }
        return state_dict
    
    def load_state(self, state_dict):
        """
        从保存的状态字典中恢复dynamics状态
        """
        # 恢复初始化参数
        if state_dict['number_steps'] is not None:
            self.number_steps = state_dict['number_steps']
        if state_dict['max_required_step'] is not None:
            self.maxRequiredStep = state_dict['max_required_step']
        self._safe_zone = state_dict['safe_zone']
        
        # 恢复计算的中间值
        self._strongest_penality_abs_idx = state_dict['strongest_penalty_abs_idx']
        
        # 恢复奖励函数相关
        self._penalty_functions_array = state_dict['penalty_functions_array']
        
        # 恢复其他可能的状态变量
        if state_dict['last_reward'] is not None:
            self._last_reward = state_dict['last_reward']
        if state_dict['last_position'] is not None:
            self._last_position = state_dict['last_position']
    


if __name__ == "__main__":
    # 测试状态保存和恢复
    dynamics_obj = dynamics(24, 0.5, 0.25)
    # 执行一些操作
    state1 = dynamics_obj.save_state()
    # 继续执行操作
    dynamics_obj.load_state(state1)
    # 验证状态是否正确恢复
    state2 = dynamics_obj.save_state()
    print(state1 == state2)
    