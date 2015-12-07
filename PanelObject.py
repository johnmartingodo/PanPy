import numpy as np
from scipy import interpolate
import Mesh

class PanelObject():
	def __init__(self, mesh):
		# Geometry data
		self.mesh = mesh
		self.mesh.calculateFaceCoordinateSystem()

		# Boundary condition type
		self.boundaryType = 'newmann'

		# Not a lifting surface by default
		self.liftingSurface = False

		# Data storage arrays for pressure, velocity and panel strength
		self.source_strength  = np.zeros(self.mesh.nrFaces)
		self.doublet_strength = np.zeros(self.mesh.nrFaces)
		self.Cp               = np.zeros(self.mesh.nrFaces)
		self.u                = np.zeros(self.mesh.nrFaces)
		self.v                = np.zeros(self.mesh.nrFaces)
		self.w                = np.zeros(self.mesh.nrFaces)
		self.U                = np.zeros(self.mesh.nrFaces)

	def calculateVelocityAndPressure(self, Uinf):
		self.Cp = np.zeros(self.mesh.nrFaces)
		self.u  = np.zeros(self.mesh.nrFaces)
		self.v  = np.zeros(self.mesh.nrFaces)
		self.w  = np.zeros(self.mesh.nrFaces)
		self.U  = np.zeros(self.mesh.nrFaces)

		# Calculate velocity due to sources
		self.u = 4*np.pi*self.mesh.face_n[:, 0]*self.source_strength
		self.v = 4*np.pi*self.mesh.face_n[:, 1]*self.source_strength
		self.w = 4*np.pi*self.mesh.face_n[:, 2]*self.source_strength

		# Calculate velocity due to doublets
		if self.liftingSurface:
			gradient = self.mesh.surfaceGradient(self.doublet_strength)

			self.u += -4*np.pi*gradient[:, 0]
			self.v += -4*np.pi*gradient[:, 1]
			self.w += -4*np.pi*gradient[:, 2]

		# Add free stream velocity
		self.u += Uinf[0]
		self.v += Uinf[1]
		self.w += Uinf[2]

		self.U = np.sqrt(self.u**2 + self.v**2 + self.w**2)

		self.Cp = 1-self.U**2/np.sqrt(Uinf[0]**2 + Uinf[1]**2 + Uinf[2]**2)

	def calculateForces(self):
		self.force = np.zeros(3)

		for i in range(self.mesh.nrFaces):
			self.force[0] += -self.Cp[i]*self.mesh.face_n[i, 0]*self.mesh.face_area[i]
			self.force[1] += -self.Cp[i]*self.mesh.face_n[i, 1]*self.mesh.face_area[i]
			self.force[2] += -self.Cp[i]*self.mesh.face_n[i, 2]*self.mesh.face_area[i]