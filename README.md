Thermodynamic Cycles Simulator — Engine Models

Overview
This interactive Streamlit application simulates thermodynamic engine cycles with a focus on Otto, Diesel, and Dual cycles using Air-Standard and Fuel-Air models. It calculates key performance metrics such as net work, thermal efficiency, mean effective pressure (MEP), and produces Pressure-Volume (P-V) and Temperature-Entropy (T-S) diagrams for visualization.

The Fuel-Air mode accounts for variable specific heats as a function of temperature and air-fuel ratio (AFR), providing a more realistic engine performance model than the simplified Air-Standard cycle.

================================================================

Features
Supports Otto, Diesel, and Dual thermodynamic cycles

Two calculation modes:

  Air-Standard (fixed properties, theoretical ideal)
  Fuel-Air (variable specific heats, realistic conditions)

Adjustable input parameters: initial pressure, temperature, compression ratio, peak temperature, calorific value, air-fuel ratio, polytropic index

Displays summary metrics: net work output, heat input, efficiency, mean effective pressure (MEP)

Plots interactive P-V, T-S, and stepwise process diagrams

Export simulation report as a downloadable PDF

Light/Dark visual themes for ease of use

File upload (CSV/Excel) for parameter input (currently limited to Air-Standard)

=====================================================================

Installation

1. Install Python 3 (preferably 3.9 or newer) from the official website:
python.org

2. Open Terminal / CMD and navigate to the folder containing app.py:

Windows:
Open Command Prompt

Type:
cd path\to\your\project

macOS / Linux:
Open Terminal

Type:
cd /path/to/your/project


3. Install the required libraries:

''pip install streamlit numpy pandas matplotlib scipy''


4. Run the application:

''streamlit run app.py''

Your browser should automatically open a page similar to:
http://localhost:8501


If it doesn’t open automatically, copy the link from the terminal and paste it into your browser manually.

5. Each time you want to run the program:

Open your terminal

Navigate to the project folder

Activate the virtual environment (if you're using one)

Run:
''streamlit run app.py ''

=======================================================================================

Usage

Select calculation mode: Air Standard or Fuel-Air.

Choose cycle type: Otto, Diesel, or Dual.

Enter parameters per your engine case (e.g., initial pressure 101.325 kPa, initial temperature 300 K, compression ratio 8).

For Fuel-Air mode, specify calorific value (kJ/kg), air-fuel ratio (AFR), and compression polytropic index.

Click Run Simulation to perform calculations.

View results in the summary metrics, tables, and diagrams tabs.

Optionally, generate a detailed PDF report via Export Report tab.

==============================================================================================

Explanation of Key Metrics

Net Work (kJ/kg): Work output per unit mass of working fluid.

Heat In (kJ/kg): Heat energy added during the combustion process.

Efficiency (%): Ratio of net work to heat input.

MEP (kPa): Mean effective pressure, representing an average pressure that produces the net work over the piston displacement.

State Points: Pressure, volume (normalized), and temperature at key cycle stages.

P-V Diagram: Illustrates pressure-volume changes through the cycle strokes.

T-S Diagram: Temperature-entropy chart approximated for idealized insight into heat and work interactions.

======================================================================================

Notes

Calculations are approximate and intended for educational and conceptual analysis.

Units follow standard SI with pressure in kPa, temperature in Kelvin, and energies in kJ per normalized mass unit.

Fuel-Air model includes variable specific heat effects as a function of temperature for realism.

Entropy is approximated for display; Fuel-Air model sets entropy to zero due to complexity.g

=======================================================================================

Contributing

Feel free to improve the code with additional cycle models, enhanced plotting, or more detailed thermodynamic properties.


