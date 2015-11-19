import numpy as np
np.set_printoptions(threshold=np.nan)
import PanelObject
import PyBEM
import FoilDatabase
import Mesh

x, y = FoilDatabase.readFile('naca0012.dat')

alpha     = 5
alpha_rad = alpha*np.pi/180
delta_y   = -1*np.sin(alpha_rad)
delta_x   = delta_y*np.sin(alpha_rad)

Asp = 5
wing = PanelObject.generateWingFrom2DProfile(x, y, Asp, 20)
flatWing = PanelObject.generateFlatWingFrom2DProfile(x, y, 10, 20, 20)

wing.mesh.rotate(np.array([0, 0, -alpha_rad]))
wing.wake_mesh.translate(0, delta_y, 0)

wing.wake_mesh.calculateFaceData()
wing.mesh.calculateFaceData()

wing.mesh.calculateFaceCoordinateSystem()
wing.wake_mesh.calculateFaceCoordinateSystem()

wing.liftingSurface = True
wing.boundaryType = 'dirichlet'



#cylinder = PanelObject.PanelObject(Mesh.importObj('cylinder.obj'))
#cylinder.boundaryType = 'newmann'
#cylinder.liftingSurface = False

sim = PyBEM.PyBEM([wing])

sim.runSimulation()
#sim.deformWake()

sim.velocityAndPressure()
#wing.calculateVelocityAndPressure(sim.Uinf)
wing.calculateForces()
print('Forces on wing:', wing.force/Asp)
CL2D = 2*np.pi*alpha_rad
CL = CL2D/(1+2/Asp)
CDi = CL**2/(np.pi*Asp)
print('Elliptic wing lift force:', CL)
print('Elliptic wing drag force:', CDi)
sim.writeResults()