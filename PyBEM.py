import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/Computation')

import PanelObject
import Computation

class PyBEM():
	def __init__(self, panelObjectList):
		self.panelObjects   = panelObjectList
		self.nrPanelObjects = len(self.panelObjects)
		self.nrCtrlPoints   = 0

		for i in range(self.nrPanelObjects):
			self.nrCtrlPoints += self.panelObjects[i].mesh.nrFaces

		# Default settings
		self.steady = True
		self.Uinf   = np.array([1, 0, 0], dtype=np.double)

	def runSimulation(self):
		startTime = time.clock()
		print('running simulation')
		
		# Algorithm for steady simulation
		if self.steady:
			self.steadyState()

		stopTime = time.clock()

		print('Simulation time:', stopTime - startTime, 's')

	def steadyState(self):
		# Global matrix system to solve, A x = b
		A = np.zeros((self.nrCtrlPoints, self.nrCtrlPoints))
		b = np.zeros(self.nrCtrlPoints)
		
		# For each panel object, calculate influence matrix
		startTime = time.clock()
		i_start = 0
		for i in range(self.nrPanelObjects):
			panelObject = self.panelObjects[i]
			ctrlMesh    = panelObject.mesh

			i_stop  = i_start + ctrlMesh.nrFaces
			j_start = 0

			for j in range(self.nrPanelObjects):
				mesh = self.panelObjects[j].mesh
				j_stop = j_start + mesh.nrFaces

				# Calculate influence on current ctrl points from current mesh
				if panelObject.boundaryType == 'dirichlet':
					if panelObject.liftingSurface:
						A_c = Computation.influenceMatrix(ctrlMesh, mesh, 'doubletPotential')
					else:
						A_c = Computation.influenceMatrix(ctrlMesh, mesh, 'sourcePotential')
				elif panelObject.boundaryType == 'newmann':
					if panelObject.liftingSurface:
						A_c = Computation.influenceMatrix(ctrlMesh, mesh, 'doubletVelocity')
					else:
						A_c = Computation.influenceMatrix(ctrlMesh, mesh, 'sourceVelocity')
				
				# Put resulting influence matrix into global matrix
				A[i_start:i_stop, j_start:j_stop] = A_c

				j_start += mesh.nrFaces

			# Compute right side of linear system
			if panelObject.boundaryType == 'dirichlet':
				if panelObject.liftingSurface:
					panelObject.source_strength = Computation.freeStreamNewmann(self.Uinf, ctrlMesh)/(4*np.pi)
					A_source                    = Computation.influenceMatrix(ctrlMesh, ctrlMesh, 'sourcePotential')
					b[i_start:i_stop]           = -np.dot(A_source, panelObject.source_strength)
				else:
					b[i_start:i_stop] = Computation.freeStreamPotential(self.Uinf, ctrlMesh)

			elif panelObject.boundaryType == 'newmann':
				b[i_start:i_stop] = Computation.freeStreamNewmann(self.Uinf, ctrlMesh)

			i_start += ctrlMesh.nrFaces

		stopTime = time.clock()
		print('Matrix assembly time:', stopTime - startTime, 's')

		# Solve system
		startTime = time.clock()
		strength = np.linalg.solve(A, b)
		stopTime = time.clock()
		print('Linear solver time:', stopTime - startTime, 's')

		# Transfer global strength values to individual panel objects
		i_start = 0
		for i in range(self.nrPanelObjects):
			i_stop = i_start + self.panelObjects[i].mesh.nrFaces

			if self.panelObjects[i].liftingSurface:
				self.panelObjects[i].doublet_strength = strength[i_start:i_stop]
			else:
				self.panelObjects[i].source_strength = strength[i_start:i_stop]

			i_start += self.panelObjects[i].mesh.nrFaces

	def velocityAndPressure(self):
		startTime = time.clock()
		# Calculate velocity
		for i in range(self.nrPanelObjects):
			# For each ctrl point on each mesh
			p = self.panelObjects[i].mesh.face_center
			u = np.zeros((self.panelObjects[i].mesh.nrFaces, 3), dtype=np.double)

			for j in range(self.nrPanelObjects):
				# Calculate influence from all panel objects
				mesh = self.panelObjects[j].mesh

				strength = self.panelObjects[j].source_strength
				u += Computation.velocity(p, strength, mesh, 'sourceVelocity')

				if self.panelObjects[j].liftingSurface:
					strength = self.panelObjects[j].doublet_strength
					u += Computation.velocity(p, strength, mesh, 'doubletVelocity')

			self.panelObjects[i].u = u[:, 0] + self.Uinf[0]
			self.panelObjects[i].v = u[:, 1] + self.Uinf[1]
			self.panelObjects[i].w = u[:, 2] + self.Uinf[2]

			self.panelObjects[i].U = np.sqrt(self.panelObjects[i].u**2 + self.panelObjects[i].v**2 + self.panelObjects[i].w**2)

			UinfMag = np.sqrt(np.sum(self.Uinf**2))
			self.panelObjects[i].Cp = 1 - self.panelObjects[i].U**2/UinfMag**2

		stopTime = time.clock()
		print('Velocity and pressure calculation time:', stopTime - startTime, 's')

	def writeResults(self):
		for i in range(self.nrPanelObjects):
			mesh = self.panelObjects[i].mesh

			mesh.addFaceData('source_strength', self.panelObjects[i].source_strength)
			mesh.addFaceData('doublet_strength', self.panelObjects[i].doublet_strength)
			mesh.addFaceData('Cp', self.panelObjects[i].Cp)
			mesh.addFaceData('u', self.panelObjects[i].u)
			mesh.addFaceData('v', self.panelObjects[i].v)
			mesh.addFaceData('w', self.panelObjects[i].w)
			mesh.addFaceData('U', self.panelObjects[i].U)

			mesh.exportVTK('panlObject{:.0f}.vtp'.format(i))