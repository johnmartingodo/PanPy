import numpy as np
np.set_printoptions(threshold=np.nan)
import PanelObject
import LiftingSurface
import PanPy
import FoilDatabase
import Mesh

import matplotlib.pyplot as plt

# ----------------------- Wing ------------------------------------
# Settings
foilFile  = 'ls417.dat'
alpha     = 0
alpha_rad = alpha*np.pi/180
Asp       = 10

# Create wing geometry from foil database
x, y = FoilDatabase.readFile(foilFile)
wing = LiftingSurface.LiftingSurface(x, y, Asp, 30)


# Rotate wing to correct angle of attack
wing.rotate(0, 0, -alpha_rad)

#wing.exportTestGeometry()

# --------------------- Cylinder ------------------------------------
# Create cylinder from mesh file
cylinder = PanelObject.PanelObject(Mesh.importObj('cylinder.obj'))

# --------------------- Run Simulation -----------------------------
sim = PanPy.PanPy([wing])

sim.steadyStateDirichlet()

#sim.velocityAndPressure()
wing.dirichletVelocityAndPressure(sim.Uinf)

# Extract pressure from mid strip
faceIndices = wing.stripFaces[15]
x = wing.mesh.face_center[faceIndices, 0]
Cp = wing.Cp[faceIndices]

plt.plot(x, -Cp)
plt.show()

wing.calculateForces()
print('Forces on wing:', wing.force/Asp)
CL2D = 2*np.pi*alpha_rad
CL = CL2D/(1+2/Asp)
CDi = CL**2/(np.pi*Asp)
print('Elliptic wing lift force:', CL)
print('Elliptic wing drag force:', CDi)
sim.writeResults()