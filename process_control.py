# Standard Imports
import sys, os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from copy import deepcopy

# Define a default response function
def linear_output_response(pv, out, slope = 0.01):
    return pv + slope * (out - 50) + np.random.normal(0, 0.05)

# Class RL Controller
class rl_controller():

    # Initialize Simulation
    def __init__(self, pv0=0, sps=np.ones(2000), pvf=linear_output_response, ql=500):
        
        # Create a Process object and store initial settings
        self.process = process(pv0, sps, pvf)
        self.pv0 = pv0
        self.sps = sps
        self.pvf = pvf

        # Screen Dimention Parameters
        self.w = 800
        self.h = 600
        self.plot_w = 650
        self.plot_h = 400
        self.axis_w = 620
        self.axis_h = 360
        self.tickmark_spacing = 100

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
        self.x_positions = ((self.axis_w/ql) * (np.arange(ql))) + (self.w - self.axis_w)/2
        
        # Episode Variables
        self.episode_complete = False
        self.episode_counter = 0
        self.total_errors = []

    # Pygame Function to Display Visual Simulations
    def run_sim(self):

        # Initialize game
        pygame.init()
        window = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("controller")

        # Setup fonts
        font = pygame.font.Font(pygame.font.get_default_font(), 18)
        
        # Start the game loop
        run = True
        while run:

            # Refresh window
            pygame.time.delay(10)
            
            # Process Reset
            if self.process.current_time == self.simulation_length:
                self.process = process(self.pv0, self.sps, self.pvf)
                self.queue_position = 0
                self.displayed_time = np.arange(self.queue_length)

            # Simulate Controller
            err = self.process.error[len(self.process.error) - 1]
            out = self.process.out[len(self.process.out) - 1] - 0.002*err**5 - 0.005*err**4 - 0.01*err**3 - 0.123*err
            self.process.run(out)

            # Draw Objects
            self.draw_window(window)
            self.draw_process(window)
            self.draw_axes(window, font)

            # Update Display
            pygame.display.update()

            # Exit on Esc
            for event in pygame.event.get():
                if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    run = False

        # End the Game
        pygame.quit()

    def draw_window(self, window):

        # Draw Frame
        pygame.draw.rect(window, (230, 230, 230), (0, 0, self.w, self.h))
        pygame.draw.rect(window, (0, 0, 0), ((self.w - self.plot_w)/2 - 10, (self.h - self.plot_h)/2 - 10, self.plot_w + 20, self.plot_h + 20))
        pygame.draw.rect(window, (255, 255, 255), ((self.w - self.plot_w)/2, (self.h - self.plot_h)/2, self.plot_w, self.plot_h))
        
    def draw_process(self, window):
        
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
        sp = self.process.sp[self.queue_position:(self.queue_position + self.queue_length)]
        out = self.process.out[self.queue_position:(len(self.process.out))]
        
        pv_rescaled = (np.ones(len(pv)) * (self.h - self.axis_h)/2) + ((((np.ones(len(pv)) * self.scale_max) - np.array(pv))/(self.scale_max - self.scale_min)) * (self.axis_h))
        sp_rescaled = (np.ones(len(sp)) * (self.h - self.axis_h)/2) + ((((np.ones(len(sp)) * self.scale_max) - np.array(sp))/(self.scale_max - self.scale_min)) * (self.axis_h))
        out_rescaled = (np.ones(len(out)) * (self.h - self.axis_h)/2) + (np.array(out)/100 * self.axis_h)

        for idx in np.arange(0, pv_rescaled.size - 1):
            pygame.draw.line(window, (10, 230, 10), (self.x_positions[idx], pv_rescaled[idx]), (self.x_positions[idx + 1], pv_rescaled[idx + 1]), 3)
        for idx in np.arange(0, sp_rescaled.size - 1):
            pygame.draw.line(window, (230, 10, 10), (self.x_positions[idx], sp_rescaled[idx]), (self.x_positions[idx + 1], sp_rescaled[idx + 1]), 3)
        for idx in np.arange(0, out_rescaled.size - 1):
            pygame.draw.line(window, (10, 10, 230), (self.x_positions[idx], out_rescaled[idx]), (self.x_positions[idx + 1], out_rescaled[idx + 1]), 3)

    def draw_axes(self, window, font):

        # Draw Vertical Axis
        pygame.draw.line(window, (0, 0, 0), ((self.w - self.axis_w)/2, (self.h - self.axis_h)/2), ((self.w - self.axis_w)/2, (self.h - self.axis_h)/2 + self.axis_h))
        
        # Draw Min Scale Text
        scale_min_txt = font.render(str(round(self.scale_min, 2)), True, (0, 0, 0))
        window.blit(scale_min_txt, dest=(self.scale_min_position[0], self.scale_min_position[1]))
        scale_min_out_txt = font.render("(0)", True, (0, 0, 255))
        window.blit(scale_min_out_txt, dest=(self.scale_min_position[0] + 50, self.scale_min_position[1]))

        # Draw Max Scale Text
        scale_max_txt = font.render(str(round(self.scale_max, 2)), True, (0, 0, 0))
        window.blit(scale_max_txt, dest=(self.scale_max_position[0], self.scale_max_position[1]))
        scale_max_out_txt = font.render("(100)", True, (0, 0, 255))
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
                    tickmark_txt = font.render(str(val), True, (0, 0, 0))
                    window.blit(tickmark_txt, dest=(xpos, self.scale_zero_position + 10))


class process():

    def __init__(self, pv0=0, sp=np.ones(2000), pvf=linear_output_response):
        
        # Specified Parameters
        self.sp = sp
        self.pv_funct = pvf

        # Init Queues
        self.current_time = 1
        self.pv = [pv0]
        self.error = [pv0 - sp[0]]
        self.out = [50]

    def run(self, o):

        # Truncate output to applicable range and update queue
        if o > 100: o = 100
        if o < 0: o = 0
        self.out.append(o)

        # Calculate PV and Error, and update their queues
        pv = self.pv_funct(self.pv[len(self.pv) - 1], o)
        sp = self.sp[self.current_time]
        self.pv.append(pv)
        self.error.append(pv - sp)
        self.current_time = self.current_time + 1