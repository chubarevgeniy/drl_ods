import numpy as np

class TicTacToeAdvanced:
    def __init__(self, n = 5, win_len = 4):
        if n < 5:
            raise('Minimal size for the grid is 5')
        if n < win_len + 1:
            raise('Length to win should be at least (size_of_grid - 1)')
        self.n = n
        self.observation_space = [n,n]
        self.action_space = {'actions': np.arange(3), 'n': n}
        self.win_len = win_len
        self.field = np.zeros((self.n,self.n),dtype=np.float32)
        self.player_index = 0
    
    def dir_check(self,i, j, i_incr, j_incr):
        s = 0
        done = False
        for k in range(self.win_len):
            if i+i_incr*k >= 0 and i+i_incr*k < self.n and j+j_incr*k >= 0 and j+j_incr*k < self.n:
                s += self.field[ i+i_incr*k,j+j_incr*k]
            else:
                break
        if np.abs(s) == self.win_len:
            done = True
        return done
    
    def check_winer(self):
        done = False
        for i in range(self.n):
            for j in range(self.n):
                #top-rigth check
                done = self.dir_check(i,j,-1,1) or done
                #rigth check
                done = self.dir_check(i,j,0,1) or done
                #bottom-rigth check
                done = self.dir_check(i,j,1,1) or done
                #bottom check
                done = self.dir_check(i,j,1,0) or done
        return done
    
    def check_done(self):
        s = 0
        for i in range(self.n):
            s += np.sum(self.field[:,i] == 0)
        if s == 0:
            return True
        else:
            return self.check_winer()
        
    def step(self, action: int, agent_index = 10):
        """
        Gets action (index of column) and returns updated state, reward, done and action_was_coorect flags.
        """
        if agent_index != 10 and agent_index != self.player_index:
            raise("Incorect agent")
        if self.check_done():
            return self.field.copy(), 0, True, True
        if np.sum(self.field[:,action] != 0) == 0:
            self.field[-1, action] = 1 if self.player_index else -1
            win = self.check_winer()
            self.player_index = (self.player_index + 1) % 2
            return self.field.copy(), win*100, self.check_done() , True
        
        possible_index = np.argmax(self.field[:,action] != 0) - 1
        if possible_index < 0:
            for i in range(1,self.n):
                if np.sum(self.field[:,(action+i)%self.n] != 0) == 0:
                    self.field[-1, (action+i)%self.n] = 1 if self.player_index else -1
                    win = self.check_winer()
                    self.player_index = (self.player_index + 1) % 2
                    return self.field.copy(), win*100, self.check_done(), False
                possible_index = np.argmax(self.field[:,(action+i)%self.n] != 0) - 1
                if possible_index >= 0:
                    self.field[possible_index, (action+i)%self.n] = 1 if self.player_index else -1
                    win = self.check_winer()
                    self.player_index = (self.player_index + 1) % 2
                    return self.field.copy(), win*100, self.check_done(), False
        else:
            self.field[possible_index, action] = 1 if self.player_index else -1
            win = self.check_winer()
            self.player_index = (self.player_index + 1) % 2
            return self.field.copy(), win*100, self.check_done(), True
        
    def reset(self):
        self.field = np.zeros((self.n,self.n),np.float32)
        self.player_index = 0
        return self.field.copy()
        
    def render(self):
        print(self.field)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            