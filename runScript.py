import numpy as np
np.set_printoptions(threshold=np.nan)
import PanelObject
import PyBEM
import FoilDatabase
import Mesh

n = 100
t = np.linspace(0, 10, n)
v = np.zeros((n, 3))

x, y = FoilDatabase.readFile('naca0018.dat')

wing = PanelObject.generateWingFrom2DProfile(x, y, 5, 20)
wing.liftingSurface = False
wing.boundaryType = 'newmann'

cylinder = PanelObject.PanelObject(Mesh.importObj('cylinder.obj'))
cylinder.boundaryType = 'newmann'
cylinder.liftingSurface = False

cube = PanelObject.PanelObject(Mesh.importObj('cube.obj'))
cube.boundaryType = 'newmann'
cube.liftingSurface = False

sim = PyBEM.PyBEM([wing])

sim.runSimulation()
sim.velocityAndPressure()
sim.writeResults()