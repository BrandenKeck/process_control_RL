# Standard Imports
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np

# Custom Learning Modules
from continuous_policy_gradient_methods import Binomial_Policy_REINFORCE

# Define a default response function
def linear_output_response(pv, out, slope = 0.001):
    return pv + slope * (out - 50) + np.random.normal(0, 0.05)

# Class RL Controller
class rl_controller():

    # Initialize Simulation
    def __init__(self, lr=1e-8, df=0.85, pv0=0, sps=np.ones(2000), pvf=linear_output_response, max_dout=0.001, tolerance=10, reward_within_tolerance=100, eql=11, sl=10, ql=500):
        
        # Create a Process object and store initial settings
        self.process = process(pv0, sps, pvf, max_dout)
        self.pv0 = pv0
        self.sps = sps
        self.pvf = pvf
        self.tol = tolerance
        self.rwt = reward_within_tolerance

        # Create Learning Objects
        self.policy_gradients = Binomial_Policy_REINFORCE(lr, df, eql, 3*sl)
        self.state = np.zeros(3*sl).tolist()
        self.state_length = 3*sl
        self.reward = 0
        self.last_action = 0

        # Hardcoded Screen Dimention Parameters
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
    def run(self):

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
            self.simulate()
            if(len(self.process.out) == 0): continue

            # Draw Objects
            self.draw_window(window)
            self.draw_process(window, fonts)
            self.draw_axes_and_text(window, fonts)

            # Update Display
            pygame.display.update()

            # Exit on Esc
            for event in pygame.event.get():
                if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    run = False

        # End the Game
        pygame.quit()

    def train(self, iterations):
        
        # Train for iterations
        while self.training_counter < iterations: self.simulate()
        self.training_counter = 0

    def explore(self, iterations, exp_factor=0.01):
        
        # Explore for iterations
        while self.training_counter < iterations: self.simulate(exp_factor=0.01)
        self.training_counter = 0

    def simulate(self, exp_factor=-1):
        
            # Simulate Controller
            if exp_factor > 0: 
                pv = self.process.pv[len(self.process.pv)-1]
                sp = self.process.sp[len(self.process.pv)-1]
                self.last_action = self.last_action + exp_factor*(sp - pv)
            else: 
                self.act()

            # Compute error and append the process
            pv, sp, err = self.process.run(self.last_action)
            
            # Process Reset
            if self.process.current_time == self.simulation_length:
                self.process = process(self.pv0, self.sps, self.pvf)
                self.queue_position = 0
                self.displayed_time = np.arange(self.queue_length)
                self.episode_complete = True
                self.episode_counter = self.episode_counter + 1
                self.training_counter = self.training_counter + 1
                print("Completed episodes: " + str(self.episode_counter))
            
            # Learn via policy gradient
            self.learn(pv, sp, err)

    def act(self):

        # Take an action based on the REINFORCE method policy
        self.last_action = self.policy_gradients.act(self.state)

    def learn(self, pv, sp, err):
        
        # Append the Error to the state queue and pop the trailing value
        self.state.append(pv)
        self.state.append(sp)
        self.state.append(err)
        while len(self.state) > self.state_length:
            self.state.pop(0)
            self.state.pop(1)
            self.state.pop(2)
        #self.state = [self.process.pv[len(self.process.pv)-1], self.process.sp[len(self.process.pv)-1]]

        # Reward is scaled to the tolerance factor
        self.reward = self.rwt * (self.tol - np.abs(err))/self.tol

        # Learn parameters for the REINFORCE method
        self.policy_gradients.learn(self.state, self.episode_complete, self.reward, self.last_action)
        if self.episode_complete: self.episode_complete = False

    def draw_window(self, window):

        # Draw Frame
        pygame.draw.rect(window, (230, 230, 230), (0, 0, self.w, self.h))
        pygame.draw.rect(window, (0, 0, 0), ((self.w - self.plot_w)/2 - 10, (self.h - self.plot_h)/2 - 10, self.plot_w + 20, self.plot_h + 20))
        pygame.draw.rect(window, (255, 255, 255), ((self.w - self.plot_w)/2, (self.h - self.plot_h)/2, self.plot_w, self.plot_h))
        
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

        for idx in np.arange(0, pv_rescaled.size - 1):
            pygame.draw.line(window, (10, 230, 10), (self.x_positions[idx], pv_rescaled[idx]), (self.x_positions[idx + 1], pv_rescaled[idx + 1]), 3)
        for idx in np.arange(0, sp_rescaled.size - 1):
            pygame.draw.line(window, (230, 10, 10), (self.x_positions[idx], sp_rescaled[idx]), (self.x_positions[idx + 1], sp_rescaled[idx + 1]), 3)
        for idx in np.arange(0, out_rescaled.size - 1):
            pygame.draw.line(window, (10, 10, 230), (self.x_positions[idx], out_rescaled[idx]), (self.x_positions[idx + 1], out_rescaled[idx + 1]), 3)
        
        pv_txt = fonts[2].render(str(round(pv[pv_rescaled.size - 1],3)), True, (10, 230, 10))
        window.blit(pv_txt, dest=(self.x_positions[pv_rescaled.size - 2] + 10, pv_rescaled[pv_rescaled.size - 2] - 6))
        sp_txt = fonts[2].render(str(round(sp[sp_rescaled.size - 1],3)), True, (230, 10, 10))
        window.blit(sp_txt, dest=(self.x_positions[sp_rescaled.size - 2] + 10, sp_rescaled[sp_rescaled.size - 2] - 6))
        out_txt = fonts[2].render(str(round(out[out_rescaled.size - 1],3)), True, (10, 10, 230))
        window.blit(out_txt, dest=(self.x_positions[out_rescaled.size - 2] + 10, out_rescaled[out_rescaled.size - 2] - 6))

    def draw_axes_and_text(self, window, fonts):

        # Draw Simulation Title Text
        title_txt = fonts[0].render("Process Control with Reinforcement Learning v0.0.1", True, (0, 0, 0))
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

        # Draw Simulation Details Text
        rwd_txt = fonts[1].render("Current Reward: " + str(self.reward), True, (0, 0, 0))
        window.blit(rwd_txt, dest=((self.w - self.plot_w)/2 + 10, self.h - (self.h - self.plot_h)/2 + 30))
        la_txt = fonts[1].render("Last Action: " + str(self.last_action), True, (0, 0, 0))
        window.blit(la_txt, dest=((self.w - self.plot_w)/2 + 10, self.h - (self.h - self.plot_h)/2 + 60))


class process():

    def __init__(self, pv0=0, sp=np.ones(2000), pvf=linear_output_response, max_dout=0.001):
        
        # Specified Parameters
        self.sp = sp
        self.pv_funct = pvf
        self.max_dout = max_dout

        # Init Queues
        self.current_time = 1
        self.pv = [pv0]
        self.out = []
        
        # Init Stored Values
        self.prev_out = 0
        self.prev_err = 0

    def run(self, o):

        # Truncate output to applicable range and update queue
        if o > self.prev_out: o = min(o+(o-self.prev_out), o+self.max_dout)
        if o < self.prev_out: o = max(o+(o-self.prev_out), o-self.max_dout)
        if o > 100: o = 100
        if o < 0: o = 0
        self.out.append(o)

        # Calculate PV and Error, and update their queues
        last_pv = self.pv[len(self.pv) - 1]
        pv = self.pv_funct(last_pv, o)
        sp = self.sp[self.current_time]

        # Update Queues
        self.pv.append(pv)
        self.current_time = self.current_time + 1

        # Return Error
        return pv, sp, (pv - sp)