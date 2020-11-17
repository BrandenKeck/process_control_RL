# Standard Imports
import pygame
import numpy as np
from copy import deepcopy

# Custom Learning Modules
from continuous_policy_gradient_methods import normal_policy_actor_critic

# Define a default response function
# Variance in slope, output, and random acceptable
def linear_output_response(pv, out, slope = 0.001):
    return pv + slope * (out - 50) #+ np.random.normal(0, 0.05)

# Class RL Controller
class rl_controller():

    # Initialize Simulation
    def __init__(self, lr=1e-8, df=0.85, 
                 pv0=0, out0=50, sps=np.ones(2000), pvf=linear_output_response, 
                 rwd_baseline=10, max_err=0.01, max_err_rwd=100, 
                 eql=11, sl=10, ql=500):
        
        # Create a Process object and store initial settings
        self.process = process(pv0=pv0, out0=out0, sp=sps, pvf=pvf)
        self.pv0 = pv0
        self.out0 = out0
        self.sps = sps
        self.pvf = pvf
        self.rwd_baseline = rwd_baseline
        self.max_err = max_err
        self.max_err_rwd = max_err_rwd

        # Create Learning Objects
        self.policy_gradients = normal_policy_actor_critic(lr, df, eql, sl)
        self.state = np.zeros(sl).tolist()
        self.state_length = sl
        self.reward = 0
        self.last_action = 0
        self.prev_last_action = 0

        # Screen Dimention Parameters
        # Parameter variance acceptable
        self.w = 800
        self.h = 600
        self.plot_w = 650
        self.plot_h = 400
        self.axis_w = 620
        self.axis_h = 360
        self.tickmark_spacing = 100
        self.queue_offset = 40

        # Position Variables
        self.scale_min = -10
        self.scale_max = 10
        self.scale_min_position = [(self.w - self.axis_w)/2 + 20, (self.h - self.axis_h)/2 + self.axis_h - 18]
        self.scale_max_position = [(self.w - self.axis_w)/2 + 20, (self.h - self.axis_h)/2]
        self.scale_zero_position = (self.h - self.axis_h)/2 - ((self.scale_max)/(self.scale_max - self.scale_min)) * (self.axis_h)

        # Episode Queue Variables
        self.queue_position = 0
        self.queue_length = ql
        self.simulation_length = sps.size
        self.displayed_time = np.arange(ql)
        self.x_positions = ((self.axis_w/(ql+self.queue_offset)) * (np.arange(ql+self.queue_offset))) + (self.w - self.axis_w)/2
        
        # Episode Variables
        self.episode_complete = False
        self.episode_counter = 0
        self.training_counter = 0

    # Pygame Function to Display Visual Simulations
    def run(self, ornstein_uhlenbeck=False, learn=True):

        # Initialize game
        pygame.init()
        window = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("controller")

        # Setup fonts
        title_font = pygame.font.Font(pygame.font.get_default_font(), 26)
        std_font = pygame.font.Font(pygame.font.get_default_font(), 18)
        mini_font = pygame.font.Font(pygame.font.get_default_font(), 12)
        fonts = [title_font, std_font, mini_font]
        
        # Start the game loop
        run = True
        while run:

            # Refresh window
            pygame.time.delay(10)

            # Run simulation
            self.simulate(ornstein_uhlenbeck, learn)
            if(len(self.process.out) == 0): continue

            # Draw Objects
            self.draw_window(window)
            self.draw_process(window, fonts)
            self.draw_axes_and_text(window, fonts, learn)

            # Update Display
            pygame.display.update()

            # Exit on Esc
            for event in pygame.event.get():
                if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    run = False

        # End the Game
        pygame.quit()

    def train(self, iterations, ornstein_uhlenbeck=True, learn=True):
        
        # Train for iterations
        while self.training_counter < iterations: self.simulate(ornstein_uhlenbeck, learn)
        self.training_counter = 0

    def simulate(self, ornstein_uhlenbeck=False, learn=True):
        
            # Act function
            self.act(ornstein_uhlenbeck, learn)

            # Compute error and append the process
            pv, sp = self.process.run(self.last_action)
            
            # Append the Error to the state queue and pop the trailing value
            self.state.append(pv - sp)
            while len(self.state) > self.state_length: self.state.pop(0)
            
            # Process Reset
            if self.process.current_time == self.simulation_length:
                self.process = process(pv0=self.process.pv[len(self.process.pv) - 1], out0=self.process.out[len(self.process.out) - 1], sp=self.sps, pvf=self.pvf)
                self.queue_position = 0
                self.displayed_time = np.arange(self.queue_length)
                self.episode_complete = True
                self.episode_counter = self.episode_counter + 1
                self.training_counter = self.training_counter + 1
                print("Completed episodes: " + str(self.episode_counter))
            
            # Learn via policy gradient
            if learn: self.learn(pv, sp)

    def act(self, ou, learn):

        # Take an action based on the REINFORCE method policy
        self.prev_last_action = self.last_action
        self.last_action = self.policy_gradients.act(self.state, ou, learn)
        if self.last_action > 100: self.last_action = 100
        if self.last_action < 0: self.last_action = 0


    def learn(self, pv, sp):

        # Reward is computed based on spec limits and additional bonuses for tight control
        self.reward = self.rwd_baseline - np.abs(pv - sp)
        #self.reward = self.rwd_baseline/np.abs(pv - sp)
        if np.abs(pv - sp) < self.max_err: self.reward = self.reward + self.max_err_rwd
        
        # Pass Reward information to the learning function
        state = deepcopy(self.state)
        self.policy_gradients.learn(state, self.episode_complete, self.reward, self.last_action)
        if self.episode_complete: self.episode_complete = False
    
    def hard_reset(self):
        self.process = process(pv0=self.pv0, out0=self.out0, sp=self.sps, pvf=self.pvf)
        self.state = np.zeros(self.state_length).tolist()
        self.queue_position = 0
        self.displayed_time = np.arange(self.queue_length)
        self.episode_complete = False
        self.training_counter = 0

    def draw_window(self, window):

        # Draw Frame
        pygame.draw.rect(window, (230, 230, 230), (0, 0, self.w, self.h))
        pygame.draw.rect(window, (0, 0, 0), ((self.w - self.plot_w)/2 - 10, (self.h - self.plot_h)/2 - 10, self.plot_w + 20, self.plot_h + 20))
        pygame.draw.rect(window, (255, 255, 255), ((self.w - self.plot_w)/2, (self.h - self.plot_h)/2, self.plot_w, self.plot_h))
    
    #Draw function
    def draw_process(self, window, fonts):
        
        # Rescale the output
        min_value = min(min(self.process.pv), min(self.process.sp))
        max_value = max(max(self.process.pv), max(self.process.sp))
        range_value = max_value - min_value
        self.scale_min = min_value - 0.1*range_value
        self.scale_max = max_value + 0.1*range_value

        if self.process.current_time > self.queue_length: 
            self.queue_position = self.process.current_time - self.queue_length
            self.displayed_time = np.arange(self.queue_position, self.queue_position + self.queue_length)
        
        pv = self.process.pv[self.queue_position:(len(self.process.pv))]
        sp = self.process.sp[self.queue_position:(len(self.process.pv))]
        out = self.process.out[self.queue_position:(len(self.process.out))]
        
        pv_rescaled = (np.ones(len(pv)) * (self.h - self.axis_h)/2) + ((((np.ones(len(pv)) * self.scale_max) - np.array(pv))/(self.scale_max - self.scale_min)) * (self.axis_h))
        sp_rescaled = (np.ones(len(sp)) * (self.h - self.axis_h)/2) + ((((np.ones(len(sp)) * self.scale_max) - np.array(sp))/(self.scale_max - self.scale_min)) * (self.axis_h))
        out_rescaled = (np.ones(len(out)) * (self.h - self.axis_h)/2) + ((1 - np.array(out)/100) * self.axis_h)

        #Drawline 0
        for idx in np.arange(0, pv_rescaled.size - 1):
            pygame.draw.line(window, (10, 230, 10), (self.x_positions[idx], pv_rescaled[idx]), (self.x_positions[idx + 1], pv_rescaled[idx + 1]), 3)
        #Drawline 1
        for idx in np.arange(0, sp_rescaled.size - 1):
            pygame.draw.line(window, (230, 10, 10), (self.x_positions[idx], sp_rescaled[idx]), (self.x_positions[idx + 1], sp_rescaled[idx + 1]), 3)
        #Drawline 2
        for idx in np.arange(0, out_rescaled.size - 1):
            pygame.draw.line(window, (10, 10, 230), (self.x_positions[idx], out_rescaled[idx]), (self.x_positions[idx + 1], out_rescaled[idx + 1]), 3)
        
        pv_txt = fonts[2].render(str(round(pv[pv_rescaled.size - 1],3)), True, (10, 230, 10))
        window.blit(pv_txt, dest=(self.x_positions[pv_rescaled.size - 2] + 10, pv_rescaled[pv_rescaled.size - 2] - 6))
        sp_txt = fonts[2].render(str(round(sp[sp_rescaled.size - 1],3)), True, (230, 10, 10))
        window.blit(sp_txt, dest=(self.x_positions[sp_rescaled.size - 2] + 10, sp_rescaled[sp_rescaled.size - 2] - 6))
        out_txt = fonts[2].render(str(round(out[out_rescaled.size - 1],3)), True, (10, 10, 230))
        window.blit(out_txt, dest=(self.x_positions[out_rescaled.size - 2] + 10, out_rescaled[out_rescaled.size - 2] - 6))

    def draw_axes_and_text(self, window, fonts, learn):

        # Draw Simulation Title Text
        # Drone control using reinforcement learning
        title_txt = fonts[0].render("Process Control with Reinforcement Learning v0.0.5", True, (0, 0, 0))
        window.blit(title_txt, dest=((self.w - self.plot_w)/2, (self.h - self.plot_h)/2 - 60))
        
        # Draw Vertical Axis
        pygame.draw.line(window, (0, 0, 0), ((self.w - self.axis_w)/2, (self.h - self.axis_h)/2), ((self.w - self.axis_w)/2, (self.h - self.axis_h)/2 + self.axis_h))
        
        # Draw Min Scale Text
        scale_min_txt = fonts[1].render(str(round(self.scale_min, 2)), True, (0, 0, 0))
        window.blit(scale_min_txt, dest=(self.scale_min_position[0], self.scale_min_position[1]))
        scale_min_out_txt = fonts[1].render("(0)", True, (0, 0, 255))
        window.blit(scale_min_out_txt, dest=(self.scale_min_position[0] + 50, self.scale_min_position[1]))

        # Draw Max Scale Text
        scale_max_txt = fonts[1].render(str(round(self.scale_max, 2)), True, (0, 0, 0))
        window.blit(scale_max_txt, dest=(self.scale_max_position[0], self.scale_max_position[1]))
        scale_max_out_txt = fonts[1].render("(100)", True, (0, 0, 255))
        window.blit(scale_max_out_txt, dest=(self.scale_max_position[0] + 50, self.scale_max_position[1]))

        if self.scale_min < 0 and self.scale_max > 0:
            
            # Draw Horizontal Axis
            self.scale_zero_position = (self.h - self.axis_h)/2 + ((self.scale_max)/(self.scale_max - self.scale_min)) * (self.axis_h)
            pygame.draw.line(window, (0, 0, 0), ((self.w - self.axis_w)/2, self.scale_zero_position), ((self.w - self.axis_w)/2 + self.axis_w, self.scale_zero_position))
            
            # Draw Tickmarks on Horizontal Axis
            axis_offset = 60
            for idx, val in enumerate(self.displayed_time[axis_offset:len(self.displayed_time)-axis_offset]):
                if val%self.tickmark_spacing == 0:
                    xpos = self.x_positions[idx + axis_offset]
                    pygame.draw.line(window, (0, 0, 0), (xpos, self.scale_zero_position - 6), (xpos, self.scale_zero_position + 6))
                    tickmark_txt = fonts[1].render(str(val), True, (0, 0, 0))
                    window.blit(tickmark_txt, dest=(xpos, self.scale_zero_position + 10))

        # Draw Current Rewards Text
        if learn: rwd_txt = fonts[1].render("Current Reward: " + str(round(self.reward,3)), True, (0, 0, 0))
        else: rwd_txt = fonts[1].render("Current Reward: <<Not Learning>>", True, (0, 0, 0))
        window.blit(rwd_txt, dest=((self.w - self.plot_w)/2 + 10, self.h - (self.h - self.plot_h)/2 + 30))
        
        # Draw Last Action Text
        la_txt = fonts[1].render("Last Action: " + str(round(self.last_action,2)), True, (0, 0, 0))
        window.blit(la_txt, dest=((self.w - self.plot_w)/2 + 10, self.h - (self.h - self.plot_h)/2 + 60))


#Process class
class process():
    
    #Initialize function
    def __init__(self, pv0=0, out0=50, sp=np.ones(2000), pvf=linear_output_response, max_dout=0.001):
        
        # Specified Parameters
        self.sp = sp
        self.pv_funct = pvf
        self.max_dout = max_dout

        # Init Queues
        self.current_time = 1
        self.pv = [pv0]
        self.out = [out0]
        
        # Init Stored Values
        self.prev_out = 0

    #Run function
    def run(self, o):

        # Append Output
        self.out.append(o)

        # Calculate PV and Error, and update their queues
        last_pv = self.pv[len(self.pv) - 1]
        pv = self.pv_funct(last_pv, o)
        sp = self.sp[self.current_time]

        # Update Queues
        self.pv.append(pv)
        self.current_time = self.current_time + 1

        # Return Error
        return pv, sp