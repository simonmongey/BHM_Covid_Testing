#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the code for the Jupyter widgets. It is not required
for the model framework. The widgets are purely for decorative purposes.
"""

#######################################################
#                    Dependencies                     #
#######################################################

from ipywidgets import widgets, Layout, Button, HBox, VBox, interactive
from IPython.core.display import display
from IPython.display import clear_output, Markdown, Latex
from collections import OrderedDict
from IPython.display import Javascript
import json
import itertools
import numpy as np
import os

# Define global parameters for parameter checks
model_solved = False

#######################################################
#          Jupyter widgets for user inputs            #
#######################################################

## This section creates the widgets that will be diplayed and used by the user
## to input parameter values.

style_mini = {'description_width': '5px'}
style_short = {'description_width': '100px'}
style_med = {'description_width': '250px'}
style_long = {'description_width': '200px'}

layout_mini =Layout(width='18.75%')
layout_50 =Layout(width='50%')
layout_med =Layout(width='70%')

widget_layout = Layout(width = '100%')

ξ_base = widgets.BoundedFloatText(
    value=0.01,
    min = 0,
    max = 1,
    step=0.001,
    disabled=False,
    description = r'Baseline quarantine rate ($\xi^u$)',
    style=style_med,
    layout = layout_med
)
A_rel = widgets.BoundedFloatText(
    value=0.5,
    min = 0,
    max = 1,
    step=0.001,
    disabled=False,
    description = r'Relative productivity of quarantined ($A_{rel}$)',
    style = style_med,
    layout =layout_med
)
d_vaccine = widgets.BoundedIntText(
    value=500,
    min = 20,
    max = 10000,
    step=1,
    disabled=False,
    description = 'Date of vaccine',
    style = style_med,
    layout = layout_med
)
rel_ρ = widgets.BoundedFloatText(
    value=1,
    min = 0.,
    max = 1,
    step=0.001,
    disabled=False,
    description = r'Relative infectiousness ($\rho^A/\rho^S$)',
    style = style_med,
    layout = layout_med
)
δ_param = widgets.BoundedIntText(
    value=6,
    min = 1,
    max = 20,
    step=1,
    disabled=False,
    description = r"Mean days to show symptoms ($1/\delta$)",
    style = style_med,
    layout = layout_med
)
ωR_param = widgets.BoundedIntText(
    value=14,
    min = 1,
    max = 60,
    step= 1,
    disabled=False,
    description = r"Mean days to recovery ($1/\omega^R$)",
    style = style_med,
    layout = layout_med
)
π_D = widgets.BoundedFloatText(
    value=0.01,
    min = 0.001,
    max = 0.99,
    step=0.001,
    disabled=False,
    description = r'Mortality rate ($\pi^D$)',
    style = style_med,
    layout = layout_med
)
R_0 = widgets.BoundedFloatText(
    value=2.5,
    min = 0.1,
    max = 8.,
    step=0.001,
    disabled=False,
    description = r'Basic reproduction number ($R_0$)',
    style = style_med,
    layout = layout_med
)

rel_λ = widgets.BoundedFloatText(
    value= 0.5,
    step= 0.001,
    min = .001,
    max = .99,
    disabled=False,
    description = r'Effectiveness of quarantine ($\lambda^Q/\lambda$)',
    style = style_med,
    layout = layout_med
)
initial_infect = widgets.BoundedIntText(
    value= 300,
    step= 1,
    min = 1,
    max = 1000,
    disabled=False,
    description = 'Number of initial infections',
    style = style_med,
    layout = layout_med
)

τ = widgets.BoundedFloatText(
    value= 0.005,
    step= 0.001,
    min = .001,
    max = .99,
    disabled=False,
    description = r'Testing rate ($\tau$)',
    style = style_med,
    layout = layout_med
)

Δ = widgets.BoundedFloatText(
    value= 0.2,
    step= 0.001,
    min = .001,
    max = .99,
    disabled=False,
    description = r'Relaxation of quarantine ($\Delta$)',
    style = style_med,
    layout = layout_med
)

τ_step = widgets.BoundedFloatText(
    value= 0.001,
    step= 0.001,
    min = .001,
    max = .1,
    disabled=False,
    description = r'Step size for $\tau$ slider',
    style = style_med,
    layout = layout_med
)

Δ_step = widgets.BoundedFloatText(
    value= 0.1,
    step= 0.001,
    min = .001,
    max = .99,
    disabled=False,
    description = r'Step size for $\Delta$ slider',
    style = style_med,
    layout = layout_med
)

τ_max = widgets.BoundedFloatText(
    value= 0.005,
    step= 0.001,
    min = .001,
    max = 1.0,
    disabled=False,
    description = r'Maximum $\overline{\tau}$ (Minimum $\underline{\tau}=0$)',
    style = style_med,
    layout = layout_med
)

Δ_min = widgets.BoundedFloatText(
    value= 0.2,
    step= 0.2,
    min = 0.0,
    max = 0.999,
    disabled=False,
    description = r'Minimum $\underline{\Delta}$ (Maximum is $\overline{\Delta}=1$)',
    style = style_med,
    layout = layout_med
)

slide_var = widgets.Dropdown(
    options = [(r'Relaxation of quarantine (Δ)',2), (r'Testing Rate (τ)',1)],
    value = 1,
    description='Slide over:',
    disabled=False,
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

def displaySlider(slide_var):
    ## This function displays the box to input households productivity
    ## if hosueholds are allowed to hold capital.
    if slide_var == 1: # slide tau
        τ.layout.display        = 'none'
        τ_max.layout.display    = 'none'
        τ_step.layout.display   = 'none'
        Δ.layout.display        = None
        Δ_min.layout.display    = None
        Δ_step.layout.display   = None
        τ.value                 = 0.005
        τ_max.value             = 0.005
        τ_step.value            = 0.001
        Δ.value                 = 0.20
        Δ_min.value             = 0.1       # Irrelevant
        Δ_step.value            = 0.1       # Irrelevant
        display(τ_max)
        display(τ_step)
        display(Δ)

    elif slide_var == 2: # slide delta
        Δ.layout.display        = 'none'
        Δ_min.layout.display    = 'none'
        Δ_step.layout.display   = 'none'
        τ.layout.display        = None
        τ_max.layout.display    = None
        τ_step.layout.display   = None
        τ.value                 = 0.005
        τ_max.value             = 0.016     # Irrelevant
        τ_step.value            = 0.001     # Irrelevant
        Δ.value                 = 0.2
        Δ_min.value             = 0.2
        Δ_step.value            = 0.2
        display(Δ_min)
        display(Δ_step)
        display(τ)

slide_varOut = widgets.interactive_output(displaySlider, {'slide_var': slide_var})

runModel = widgets.Button(
    description='Run model',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

displayPlotPanel = widgets.Button(
    description='Show panel chart',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

box_layout       = Layout(width='100%', flex_flow = 'row')#, justify_content='space-between')
box_layout_wide  = Layout(width='100%', justify_content='space-between')
box_layout_small = Layout(width='50%')

Quarantine_box = VBox([widgets.Label(value="1. Quarantine parameters"), ξ_base, \
                       rel_λ, A_rel], layout = box_layout_wide)
Timing_box = VBox([widgets.Label(value="2. Timing parameters"), d_vaccine, \
                       δ_param, ωR_param], layout = box_layout_wide)
Spread_box = VBox([widgets.Label(value="3. Contagion parameters"), R_0, rel_ρ], \
layout = box_layout_wide)
Disease_box = VBox([widgets.Label(value="4. Disease parameters"), initial_infect, π_D], \
layout = box_layout_wide)
slide_var_box = VBox([widgets.Label(value="5. Policy parameters"), slide_var, slide_varOut], \
layout = box_layout_wide)
# slide_var_box1 = VBox([widgets.Label(value="Plotting Parameters"), slide_var], \
# layout = box_layout_small)
# slide_var_box2 = VBox([widgets.Label(value="Plotting Parameters"), slide_varOut], \
# layout = box_layout_small)


line1      = HBox([Quarantine_box, Timing_box], layout = box_layout)
line2      = HBox([Disease_box, Spread_box], layout = box_layout)
line3      = HBox([slide_var_box], layout = box_layout)
paramsPanel = VBox([line1, line2, line3])
paramsPanel2d = VBox([line1, line2])
run_box = VBox([widgets.Label(value="Execute Model"), runModel, \
                displayPlotPanel])



#######################################################
#                      Functions                      #
#######################################################

def runModelFn(b):
    ## This is the function triggered by the runModel button.
    clear_output() ## clear the output of the existing print-out
    display(run_box) ## after clearing output, re-display buttons
    global model_solved
    print("Solving the model...")
    display(Javascript("Jupyter.notebook.execute_cells([7])"))
    model_solved = True

def displayPlotPanelFn(b):
    if model_solved:
        print("Showing plots.")
        display(Javascript("Jupyter.notebook.execute_cells([8])"))
    else:
        print("You must run the model first.")

#######################################################
#                 Configure buttons                   #
#######################################################

selectedMoments = []

runModel.on_click(runModelFn)
displayPlotPanel.on_click(displayPlotPanelFn)
